///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////

//Modified for anti-spoofing 
//Copyright @2018. All rights reserved.
//Authors:  anaitsat@campus.technion.ac.il  (Alexander Naitsat)

// FaceTrackingVidMulti.cpp : Defines the entry point for the multiple face tracking console application.
#include "LandmarkCoreIncludes.h"

#include "../../FaceAnalyser/include/FaceAnalyser.h"
#include "../../FaceAnalyser/include/GazeEstimation.h"


#include <fstream>
#include <sstream>
#include <regex>
#include <string>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl
//extern int LandmarkDetector::selected_landmark;
int selected_landmark_num = -1;
//std::set<int> five_landmarks = {2,8,14,33,36}; //5 points for planrity check 
 //std::vector<int> five_landmarks = { 45,36,30,48,54 }; //5 points for planrity check 

std::vector<int> five_landmarks = { 30,48,36,45,54}; //5 points for planrity check 

std::vector<cv::Point> five_landmarks_coord;
#define TEST_FRAME 8


#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

typedef vector<cv::Point> PointVector;
//typedef cv::Mat_<cv::Point> PointMat;
typedef cv::Mat PointMat;
cv::Scalar status_color = CV_RGB(0, 0, 255);

//parameters for spoofing detection calculations
//double pyramid_height = 0.4, pyramid_num = 1;
double pyramid_height = 1, pyramid_num = 5;
int pyramid_ind = 43;//18;
double s_threshold = 0.65, g_threshold = 0.8;

cv::Point3f RaySphereIntersect(cv::Point3f rayOrigin, cv::Point3f rayDir, cv::Point3f sphereOrigin, float sphereRadius) {

	float dx = rayDir.x;
	float dy = rayDir.y;
	float dz = rayDir.z;
	float x0 = rayOrigin.x;
	float y0 = rayOrigin.y;
	float z0 = rayOrigin.z;
	float cx = sphereOrigin.x;
	float cy = sphereOrigin.y;
	float cz = sphereOrigin.z;
	float r = sphereRadius;

	float a = dx*dx + dy*dy + dz*dz;
	float b = 2 * dx*(x0 - cx) + 2 * dy*(y0 - cy) + 2 * dz*(z0 - cz);
	float c = cx*cx + cy*cy + cz*cz + x0*x0 + y0*y0 + z0*z0 + -2 * (cx*x0 + cy*y0 + cz*z0) - r*r;

	float disc = b*b - 4 * a*c;

	float t = (-b - sqrt(b*b - 4 * a*c)) / 2 * a;

	// This implies that the lines did not intersect, point straight ahead
	if (b*b - 4 * a*c < 0)
		return cv::Point3f(0, 0, -1);

	return rayOrigin + rayDir * t;
}

cv::Point3f GetPupilPosition(cv::Mat_<double> eyeLdmks3d) {

	eyeLdmks3d = eyeLdmks3d.t();

	cv::Mat_<double> irisLdmks3d = eyeLdmks3d.rowRange(0, 8);

	cv::Point3f p(mean(irisLdmks3d.col(0))[0], mean(irisLdmks3d.col(1))[0], mean(irisLdmks3d.col(2))[0]);
	return p;
}

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

void NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<double> >& face_detections)
{

	// Go over the model and eliminate detections that are not informative (there already is a tracker there)
	for(size_t model = 0; model < clnf_models.size(); ++model)
	{

		// See if the detections intersect
		cv::Rect_<double> model_rect = clnf_models[model].GetBoundingBox();
		
		for(int detection = face_detections.size()-1; detection >=0; --detection)
		{
			double intersection_area = (model_rect & face_detections[detection]).area();
			double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;

			// If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
			if( intersection_area/union_area > 0.5)
			{
				face_detections.erase(face_detections.begin() + detection);
			}
		}
	}
}

//anti-spoofing
double  process_frame(vector<LandmarkDetector::CLNF>& clnf_models, cv::Mat& captured_image,
	cv::Mat& disp_image,
	cv::VideoCapture& video_capture, vector<bool>&  active_models,
	vector<LandmarkDetector::FaceModelParameters>& det_parameters,
	float& fx, float& fy, float& cx, float& cy, int frame_count,
	int device, std::vector<cv::Point>& callibration_pnt,
	std::vector<cv::Point>& callibration_pnt_planar,
	//PointMat&  callibrationPntMat,
	vector<PointVector>&  callibrationPnts,
	cv::Vec3d& euler_angles,
	const string&  status_str = "")
{
	    double fps = 10;
		
		double cp1_invariant;
		cv::Mat_<float> depth_image;
		cv::Mat_<uchar> grayscale_image;

		//cv::Mat disp_image = captured_image.clone();
		disp_image = captured_image.clone();
		
		if (captured_image.channels() == 3)
		{
			cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);//converting current frame image to a graysacel image in 'grayscale_image' matrix				
		}
		else
		{
			grayscale_image = captured_image.clone();
		}

		vector<cv::Rect_<double> > face_detections;


		bool all_models_active = true;
		for (unsigned int model = 0; model < clnf_models.size(); ++model)
		{
			if (!active_models[model])
			{
				all_models_active = false;
			}
		}

		// Get the detections (every 8th frame and when there are free models available for tracking)
		//Assigns to 'face_detection' rectangles where faces were detected
		if (frame_count % TEST_FRAME == 0 && !all_models_active)
		{
			if (det_parameters[0].curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
			{
				vector<double> confidences, confidences1;
				LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, clnf_models[0].face_detector_HOG, confidences);
			}
			else
			{
				LandmarkDetector::DetectFaces(face_detections, grayscale_image, clnf_models[0].face_detector_HAAR);
			}

		}

		// Keep only non overlapping detections (also convert to a concurrent vector
		//Remove from face_detections rectangles 
		NonOverlapingDetections(clnf_models, face_detections);
		
		vector<tbb::atomic<bool> > face_detections_used(face_detections.size());

		// Go through every model and update the tracking
		tbb::parallel_for(0, (int)clnf_models.size(), [&](int model) {
			//for(unsigned int model = 0; model < clnf_models.size(); ++model)
			//{

			bool detection_success = false;

			// If the current model has failed more than 4 times in a row, remove it
			if (clnf_models[model].failures_in_a_row > 4)
			{
				active_models[model] = false;
				clnf_models[model].Reset();

			}

			// If the model is inactive reactivate it with new detections
			if (!active_models[model])
			{

				for (size_t detection_ind = 0; detection_ind < face_detections.size(); ++detection_ind)
				{
					// if it was not taken by another tracker take it (if it is false swap it to true and enter detection, this makes it parallel safe)
					if (face_detections_used[detection_ind].compare_and_swap(true, false) == false)
					{

						// Reinitialise the model
						clnf_models[model].Reset();

						// This ensures that a wider window is used for the initial landmark localisation
						clnf_models[model].detection_success = false;
						detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_detections[detection_ind], clnf_models[model], det_parameters[model]);

						// This activates the model
						active_models[model] = true;

						// break out of the loop as the tracker has been reinitialised
						break;
					}

				}
			}
			else
			{
				// The actual facial landmark detection / tracking
				detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, clnf_models[model], det_parameters[model]);
			}
		});

		// Go through every model and visualise the results
		for (size_t model = 0; model < clnf_models.size(); ++model)
		{
			// Visualisation of  the results
			// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
			double detection_certainty = clnf_models[model].detection_certainty;

			double visualisation_boundary = -0.1;

			// Only draw if the reliability is reasonable, the value is slightly ad-hoc
			if (detection_certainty < visualisation_boundary)
			{
				LandmarkDetector::Draw(disp_image, clnf_models[model]);
				//face landmarks were drawn and global vector 'five_landmark_coord' were updated  
				//double M123 = LandmarkDetector::getM(five_landmarks_coord, 1, 2, 3);
				cp1_invariant = LandmarkDetector::get_cp1(five_landmarks_coord);

				if (detection_certainty > 1)
					detection_certainty = 1;
				if (detection_certainty < -1)
					detection_certainty = -1;

				detection_certainty = (detection_certainty + 1) / (visualisation_boundary + 1);

				// Gaze tracking, absolute gaze direction
				cv::Point3f gazeDirection0(0, 0, -1);
				cv::Point3f gazeDirection1(0, 0, -1);


				bool left_eye = true;
				cv::Vec6d headPose = LandmarkDetector::GetPoseCamera(clnf_models[model], fx, fy, cx, cy);
				cv::Vec3d eulerAngles(headPose(3), headPose(4), headPose(5));
				cv::Matx33d rotMat = LandmarkDetector::Euler2RotationMatrix(eulerAngles);
				// check for pose change
				//man(angles_20) - min(angles_20)
				int part = -1;
				//going over detected face models looking for specific eye landmarks
				for (size_t i = 0; i < clnf_models[model].hierarchical_models.size(); ++i)
				{
					if (left_eye && clnf_models[model].hierarchical_model_names[i].compare("left_eye_28") == 0)
					{
						part = i;
					}
					if (!left_eye && clnf_models[model].hierarchical_model_names[i].compare("right_eye_28") == 0)
					{
						part = i;
					}
				}

				if (part == -1)
				{
					std::cout << "Couldn't find the eye model, something wrong" << std::endl;
				}
				//Using detected eye landmarks compute approximate gaze direction 
				cv::Mat eyeLdmks3d = clnf_models[model].hierarchical_models[part].GetShape(fx, fy, cx, cy);
				cv::Point3f pupil = GetPupilPosition(eyeLdmks3d);
				cv::Point3f rayDir = pupil / norm(pupil);

				cv::Mat faceLdmks3d = clnf_models[model].GetShape(fx, fy, cx, cy);
				faceLdmks3d = faceLdmks3d.t();
				cv::Mat offset = (cv::Mat_<double>(3, 1) << 0, -3.50, 0);
				int eyeIdx = 1;
				if (left_eye)
				{
					eyeIdx = 0;
				}

				cv::Mat eyeballCentreMat = (faceLdmks3d.row(36 + eyeIdx * 6) + faceLdmks3d.row(39 + eyeIdx * 6)) / 2.0f + (cv::Mat(rotMat)*offset).t();

				cv::Point3f eyeballCentre = cv::Point3f(eyeballCentreMat);

				cv::Point3f gazeVecAxis = RaySphereIntersect(cv::Point3f(0, 0, 0), rayDir, eyeballCentre, 12) - eyeballCentre;

				gazeDirection0 = gazeVecAxis / norm(gazeVecAxis);



				// A rough heuristic for box around the face width
				int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

				// Work out the pose of the head from the tracked model
				cv::Vec6d pose_estimate = LandmarkDetector::GetCorrectedPoseWorld(clnf_models[model], fx, fy, cx, cy);

				// Draw it in reddish if uncertain, blueish if certain
				LandmarkDetector::DrawBox(disp_image, pose_estimate, cv::Scalar((1 - detection_certainty)*255.0, 0, detection_certainty * 255),
					                      thickness, fx, fy, cx, cy, callibration_pnt, callibration_pnt_planar);

				euler_angles[0] = pose_estimate[3];
				euler_angles[1] = pose_estimate[4];
				euler_angles[2] = pose_estimate[5];


				vector<cv::Point> pyramid_2d_points;
		
				
				LandmarkDetector::Pyramid   P(0.5, 0.5, -0.5, -0.5, 0, 2);
				int ind = 0;

				LandmarkDetector::MatrixD M = {
							{0.3, 0.3, 0}, //button left
							{0.45, -0.4,0}, //top left
							{-0.45,-0.4,0}, //top right
							{-0.3,0.3, 0}, //button right
							{ 0,  0,   0}
							//{x  y    z}
				};
				const double eps = 0.00001;
				for (float z = -0.2; z <= 0.2+eps; z += 0.1)
				for (float j = -0.1; j <= 0.1 + eps; j += 0.1)
					for (float i = 0; i <= pyramid_height + eps; i += pyramid_height / pyramid_num, ind++)
					{
						bool is_draw = (ind == pyramid_ind);
						/*if (abs(i- 0.2) < 0.00001 && abs(j)< 0.0001 && abs(z) < 0.001) {
							std::cout << "\n Pyramid_index ===" << ind;
						}*/
						callibrationPnts.push_back(PointVector());

						//P.h = -i;
						M[4][2] = -i;
						M[4][1] =  j;

						for (int basis_pnt = 0; basis_pnt < 4;basis_pnt++)	M[basis_pnt][2] = z;
						
						
						LandmarkDetector::DrawFivePoints(disp_image, pose_estimate, cv::Scalar(150, 150, 150),
							2, fx, fy, cx, cy, callibrationPnts.back(), M, is_draw);
					}

				if (false) {
					cout << "\n callibrationPnts=";
					for (auto pi : callibrationPnts)
						cout << "\n" << pi;

					std::cout << "\n cp=" << LandmarkDetector::get_cp1(callibrationPnts[pyramid_ind]) << std::endl;

					cout << "\n callibration flat=\n" << callibration_pnt_planar;
					std::cout << "\n cp=" << LandmarkDetector::get_cp1(callibration_pnt_planar) << std::endl;
				}
			}
		}


		bool is_print_anaotations = false;
		int num_active_models = 0;

		for (size_t active_model = 0; active_model < active_models.size(); active_model++)
		{
			if (active_models[active_model])
			{
				num_active_models++;
			}
		}

		char active_m_C[255];
		sprintf(active_m_C, "%d", num_active_models);
		string active_models_st("Active models:");
		active_models_st += active_m_C;
		if (is_print_anaotations)
			cv::putText(disp_image, active_models_st, cv::Point(10, 80), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);

		string landmark_select_str = string("slected:") + to_string(selected_landmark_num);//me
		if (is_print_anaotations)
			cv::putText(disp_image, landmark_select_str, cv::Point(10, 90), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);

	   std::vector<std::string> sentences;
		boost::split(sentences, status_str, boost::is_any_of("\n"), boost::token_compress_on);

		double lineNum = 30, font_size = 0.8;
		int thikness = 1;
		auto lineCol = CV_RGB(255, 255, 0);
		for (auto str: sentences) {
			if (!str.empty())
				cv::putText(disp_image, str, cv::Point(15, lineNum), CV_FONT_HERSHEY_SIMPLEX,font_size, lineCol, thikness, CV_AA);
			    //lineCol = CV_RGB(0, 0, 255);
				lineCol = status_color;
				font_size = 2;
				lineNum += 15*(font_size/0.8);
				thikness = 4;
		}


		if (!det_parameters[0].quiet_mode)
		{
			if (device == 0) {
				cv::namedWindow("tracking_result_camera1", 1);
				cv::imshow("tracking_result_camera1", disp_image);
			} else if(device == 1) {
				cv::namedWindow("tracking_result_camera2", 1);
				cv::imshow("tracking_result_camera2", disp_image);
			}

			if (!depth_image.empty())
			{
				// Division needed for visualisation purposes
				imshow("depth", depth_image / 2000.0);
			}
		}

		video_capture >> captured_image;
		return cp1_invariant; 
}
	

int load_avi_files(string file_with_name_list, vector<string>& files, vector<bool>& is_G_statuses) {
	std::ifstream list_fh(file_with_name_list);
	if (! list_fh.is_open()) {
		std::cout << "\n Cannot open " << file_with_name_list;
		return 0;
	}
	std::string file_name;
	regex re_is_S("\_S\_");

	while (std::getline(list_fh, file_name)) {
		files.push_back(file_name);
		
		std::match_results<std::string::iterator> match;
		is_G_statuses.push_back(!std::regex_search(file_name.begin(), file_name.end(), match, re_is_S));
	}
	files.push_back(""); //empty string at the end for termination 
	is_G_statuses.push_back(true);
	list_fh.close();
	return files.size();
}


#define RESET_MODEL_PER_SESSION 1
int main(int argc, char **argv)
{

	bool is_record_video=true; //if true each recoridng is saved in avi file for each camera 
	bool is_load_video_from_file=false;//if true each, then each input as a  separate avi file (name of files are in ".\video_camkera0,1.txt")
	vector<string> arguments = get_arguments(argc, argv);

	if (std::find(arguments.begin(), arguments.end(), "-record_video") != arguments.end())
		is_record_video = true;
	else 
		is_record_video = false;


	if (std::find(arguments.begin(), arguments.end(), "-load_video") != arguments.end())
		is_load_video_from_file = true;
	else
		is_load_video_from_file = false;

	bool hide_result_annotations = true, is_print_angles = false;

	if (std::find(arguments.begin(), arguments.end(), "-annotation") != arguments.end())
		hide_result_annotations = false;
	else
		hide_result_annotations = true;

	if (std::find(arguments.begin(), arguments.end(), "-show_angle") != arguments.end())
		is_print_angles = true;
	else
		is_print_angles = false;


	// Some initial parameters that can be overwtitten  from the command line	
	vector<string> files,files1, tracked_videos_output, dummy_out;
	vector<bool> is_G_statuses, is_G_statuses1;

	// By default try webcam 0
	int device = 0, device1 = 1;//0=native laptop camera, 1=external camera (Real scene working well)
	int prev_device = device;
	// cx and cy aren't necessarilly in the image center, so need to be able to overwrite it (start with unit values  and init them if the flag is  not set)
	float fx = 600, fy = 600, cx = 0, cy = 0;

	LandmarkDetector::FaceModelParameters det_params(arguments);//in this mode there is no spedcial arguments, FaceModelParams are 'default'
	det_params.use_face_template = true;
	// disable model re-initialization
	det_params.reinit_video_every = -1;

	det_params.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;

	vector<LandmarkDetector::FaceModelParameters> det_parameters, det_parameters1;
	det_parameters.push_back(det_params);
	det_parameters1.push_back(det_params);

	// Get the input/output file parameters
	bool u;
	string output_codec;
	//below function has effect only if arguments contain specific flags for setting each variable, e.g. "-f video_file_name"
	LandmarkDetector::get_video_input_output_params(files, dummy_out, tracked_videos_output, u, output_codec, arguments);
	// Get camera parameters: has an impact if arguments contain additional flags, such as: '-fx','-fy','-device' and etc.
	LandmarkDetector::get_camera_params(device, fx, fy, cx, cy, arguments);

	// tracking modules
	vector<LandmarkDetector::CLNF> clnf_models,clnf_models1;//this contains all data, acquired during by face detection  unit
	vector<bool> active_models, active_models1;

	int num_faces_max = 1;// initally it was set to 4 

	LandmarkDetector::CLNF clnf_model(det_parameters[0].model_location), clnf_model1(det_parameters1[0].model_location);
	clnf_model.face_detector_HAAR.load(det_parameters[0].face_detector_location);
	clnf_model.face_detector_location = det_parameters[0].face_detector_location;
	clnf_models.reserve(num_faces_max);
	clnf_models.push_back(clnf_model);
	active_models.push_back(false);

	clnf_model1.face_detector_HAAR.load(det_parameters1[0].face_detector_location);
	clnf_model1.face_detector_location = det_parameters1[0].face_detector_location;
	clnf_models1.reserve(num_faces_max);
	clnf_models1.push_back(clnf_model1);
	active_models1.push_back(false);



	for (int i = 1; i < num_faces_max; ++i)
	{
		clnf_models.push_back(clnf_model);
		active_models.push_back(false);
		det_parameters.push_back(det_params);

		clnf_models1.push_back(clnf_model1);
		active_models1.push_back(false);
		det_parameters1.push_back(det_params);
	}

	// If multiple video files are tracked, use this to indicate if we are done
	bool done = false;
	int f_n = -1;

	// If cx (optical axis center) is not defined, then  use center = the image_size/2
	bool cx_undefined = false;
	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}

	five_landmarks_coord.resize(5);
	bool measure_invarinats = false, video_record_measurings = false;
	int frames_measure_num = 8;
	vector<cv::Point> bbox5point, bbox5point1; //bbox 5 corners for cross ratio callibration  
	vector<cv::Point> bbox5point_planar, bbox5point1_planar; //bbox 5 corners for cross ratio callibration  
	//double s_threshold = 0.2, g_threshold = 0.3;
	bool is_genuine_status = true;
	string result_status = "U";
	cv::Vec3d euler_angles(0, 0, 0), euler_angles1(0, 0, 0);


	if (is_load_video_from_file) {
		load_avi_files(".\\video_names_camera0.txt", files,is_G_statuses);
		load_avi_files(".\\video_names_camera1.txt", files1, is_G_statuses1);
	}

	int test_num = 0;

	while (!done) // this is not a for loop  (we might be reading from a webcam)
	{
		double cp1_invariant = 0;

		string current_file, current_file1;

		// multiple video files can be specified by input arguments
		if (files.size() > 0)
		{
			f_n++;
			current_file  = files[f_n];
			current_file1 = files1[f_n];
			is_genuine_status = is_G_statuses[f_n];
		}

		// Do some grabbing
		cv::VideoCapture video_capture, video_capture1;
		cv::VideoWriter outputVideo, outputVideo1;

		if (current_file.size() > 0)
		{
			INFO_STREAM("Attempting to read from file: " << current_file);
			video_capture = cv::VideoCapture(current_file);

			INFO_STREAM("Attempting to read from file: " << current_file);
			video_capture1 = cv::VideoCapture(current_file1);
			measure_invarinats = true; //measuring cross-ratio from the beggining
			is_record_video = false;

		}
		else
		{
			is_load_video_from_file = false;
			INFO_STREAM("Attempting to capture from device: " << device);
			video_capture = cv::VideoCapture(device);

			// Read a first frame (often, it's  empty)
			cv::Mat captured_image;
			video_capture >> captured_image;


			INFO_STREAM("Attempting to capture from device: " << device1);
			video_capture1 = cv::VideoCapture(device1);

			// Read a first frame (often, it's  empty)
			cv::Mat captured_image1;
			video_capture1 >> captured_image1;
		}

		if (!video_capture.isOpened())
		{
			FATAL_STREAM("Failed to open video source");
			return 1;
		}
		else INFO_STREAM("Device or file opened");

		double  video_fps = video_capture.get(CV_CAP_PROP_FPS),
				video_fps1 = video_capture1.get(CV_CAP_PROP_FPS);
		if (video_fps != video_fps || video_fps <= 0) {
			WARN_STREAM("FPS of the video 0  file cannot be determined, assuming 30");
			video_fps = 30;
		}
		if (video_fps1 != video_fps1 || video_fps1 <= 0) {
			WARN_STREAM("FPS of the video 1  file cannot be determined, assuming 30");
			video_fps1 = 30;
		}

		int frame_width   = (int)video_capture.get(CV_CAP_PROP_FRAME_WIDTH),
			frame_height  = (int)video_capture.get(CV_CAP_PROP_FRAME_HEIGHT),
			frame_width1  = (int)video_capture1.get(CV_CAP_PROP_FRAME_WIDTH),
			frame_height1 = (int)video_capture1.get(CV_CAP_PROP_FRAME_HEIGHT);


		cv::Mat captured_image, captured_image1;
		video_capture >> captured_image;	//saving current frame to 'capture_image' matrix
		video_capture1 >> captured_image1;	//saving current frame to 'capture_image' matrix


											// If optical centers are not defined just use center of image
		if (cx_undefined)
		{
			cx = captured_image.cols / 2.0f;
			cy = captured_image.rows / 2.0f;
		}

		int frame_count = 0, frame_count4out_video = 0, additional_video_frame = 0, additional_video_num = 18;

		// saving the videos
		cv::VideoWriter writerFace;
		if (!tracked_videos_output.empty())
		{
			try
			{
				writerFace = cv::VideoWriter(tracked_videos_output[f_n], CV_FOURCC(output_codec[0], output_codec[1], output_codec[2], output_codec[3]), 30, captured_image.size(), true);
			}
			catch (cv::Exception e)
			{
				WARN_STREAM("Could not open VideoWriter, OUTPUT FILE WILL NOT BE WRITTEN. Currently using codec " << output_codec << ", try using an other one (-oc option)");
			}
		}

		// timing measurements 
		int64 t1, t0 = cv::getTickCount();
		double fps = 10;
		int frame_num,frame_num1;
		double angles_20[20];
		INFO_STREAM("Starting tracking");
		double fivePinvariant=0, fivePinvariant1 = 0; //five point invariants 
		//while(!captured_image.empty() prev_device == device)// main loop running  while camera campture images
		vector<double> FiveInvariants, FiveInvariants1, bboxInvarinats, bboxInvarinats1, bboxInvarinats_planar, bboxInvarinats1_planar;
		string status_str, status_str1;
		double cr_threshold = 0.5,//threshold to determine if CR (cross-ratio) is the same for two camera, or it's relative to  bbox center callibration (head in (NHC))
			planar_threshold = 5;
		
		int measured_frames = 0;
		vector<vector<double>> callibPntsInvDiff;
		LandmarkDetector::set_draw_params();
		
		ofstream is_genuine_file;
		if (is_record_video)
			is_genuine_file.open("./videos/is_genuine.txt", ios_base::app);
	

		while (!captured_image.empty() && !captured_image1.empty())// main loop, it's running  while camera campture images
		{
			bbox5point.clear();
			bbox5point1.clear();
			bbox5point_planar.clear();
			bbox5point1_planar.clear();

			vector<PointVector> callibrationPnts, callibrationPnts1;
			
			cv::Mat disp_image, disp_image1;
			fivePinvariant = process_frame(clnf_models, captured_image, disp_image, video_capture,
									    	active_models, det_parameters,
											fx, fy, cx, cy, frame_count,device, bbox5point,
											bbox5point_planar, callibrationPnts, euler_angles,
											status_str);
			//fivePinvariant = LandmarkDetector::get_cp1(five_landmarks_coord);

			fivePinvariant1 = process_frame(clnf_models1, captured_image1, disp_image1,video_capture1,
											active_models1, det_parameters1,
											fx, fy, cx, cy, frame_count,device1,bbox5point1,
											bbox5point1_planar, callibrationPnts1, euler_angles1,
											status_str1);
			
			int calinb_num = callibrationPnts.size();
			vector<double> callibPntsInv(calinb_num), callibPntsInv1(calinb_num), callibInvDiffSum(calinb_num);
			

			if (is_record_video && (measure_invarinats || additional_video_frame >0)) {
				if (frame_count4out_video == 0) {
					string status = (is_genuine_status )? "G":"S";
					string video_file_name = "./videos/Camera0_"+ status  + "_" + std::to_string(test_num) + ".avi",
					  	  video_file_name1 = "./videos/Camera1_"+ status +"_" + std::to_string(test_num) + ".avi";

					cv::Size frame_size = cv::Size(frame_width, frame_height),
						frame_size1 = cv::Size(frame_width1, frame_height1);
					std::cout << "\n video_fps=" << video_fps << ", frame_size=" << frame_size;
					std::cout << "\n video_fps1=" << video_fps1 << ", frame_size1=" << frame_size1;

					outputVideo.open(video_file_name, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), video_fps, frame_size, true);
					outputVideo1.open(video_file_name1, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), video_fps1, frame_size1, true);
					if (!outputVideo.isOpened())
					{
						FATAL_STREAM("Could not open the output video for write: " + video_file_name) ;
						return 1;
					}

					if (!outputVideo1.isOpened())
					{
						FATAL_STREAM("Could not open the output video for write: " + video_file_name1);
						return 1;
					}
				}
				outputVideo << captured_image;
				outputVideo1 << captured_image1;
				frame_count4out_video++;


				if (additional_video_frame > 0)
					additional_video_frame++;
				if (additional_video_frame > 20) {
					additional_video_frame = 0;
					frame_count4out_video = 0;
				}
			}
			
			if (is_print_angles) {
				status_str = "(" + std::to_string(euler_angles[0]) + ","
					+ std::to_string(euler_angles[1]) + ","
					+ std::to_string(euler_angles[2]) + ")";

				status_str1 = "(" + std::to_string(euler_angles1[0]) + ","
					+ std::to_string(euler_angles1[1]) + ","
					+ std::to_string(euler_angles1[2]) + ")";

			}

			if (measure_invarinats && (frame_count % TEST_FRAME == 0))
			{
				FiveInvariants.push_back(fivePinvariant);
				FiveInvariants1.push_back(fivePinvariant1);

				bboxInvarinats.push_back(LandmarkDetector::get_cp1(bbox5point));
				bboxInvarinats_planar.push_back(LandmarkDetector::get_cp1(bbox5point_planar));

				bboxInvarinats1.push_back(LandmarkDetector::get_cp1(bbox5point1));
				bboxInvarinats1_planar.push_back(LandmarkDetector::get_cp1(bbox5point1_planar));

				
				vector<double> callibPntsInvDiff_frame(callibrationPnts.size());

				for (int i = 0; i < callibrationPnts.size(); i++) {

					callibPntsInv[i]  = LandmarkDetector::get_cp1(callibrationPnts[i]) ;
					callibPntsInv1[i] = LandmarkDetector::get_cp1(callibrationPnts1[i]);
					callibPntsInvDiff_frame[i] = abs(callibPntsInv1[i] - callibPntsInv[i]);
				}
				callibPntsInvDiff.push_back(callibPntsInvDiff_frame);

				if (FiveInvariants.size() >= frames_measure_num) { //end of the session
					measure_invarinats = false;
					additional_video_frame = 1;
					double inv_sum = 0, inv_sum1 = 0, mean_inv_diff = 0,
						   bbox_sum = 0, bbox_sum1 = 0, bbox_diff_sum = 0,
						   bbox_diff_sum_panar, bbox_sum_planar, bbox_sum1_planar,
						   bbox_diff_sum_planar = 0, face_diff_sum = 0;
						   callibInvDiffSum.assign(callibInvDiffSum.size(), 0);

					for (int i = 0; i < frames_measure_num; i++) {
						inv_sum += FiveInvariants[i];
						inv_sum1 += FiveInvariants1[i];

						bbox_sum += bboxInvarinats[i];
						bbox_sum1 += bboxInvarinats1[i];

						bbox_sum_planar  += bboxInvarinats_planar[i];
						bbox_sum1_planar += bboxInvarinats1_planar[i];

						bbox_diff_sum += abs(bboxInvarinats1[i] - bboxInvarinats[i]);
						face_diff_sum += abs(FiveInvariants1[i] - FiveInvariants[i]);
						bbox_diff_sum_planar += abs(bboxInvarinats1_planar[i] - bboxInvarinats_planar[i]);

						for(int j=0; j<callibrationPnts.size(); j++)
								callibInvDiffSum[j] += callibPntsInvDiff[i][j];
					}
						

					mean_inv_diff = std::abs(inv_sum - inv_sum1) / frames_measure_num;
					std::cout << "\n ====> Five point invariant mean:" << inv_sum / frames_measure_num << ", " << inv_sum1 / frames_measure_num << std::endl;
					

					double bbox_cr_mean = bbox_sum / (double)frames_measure_num, bbox_cr_mean1 = bbox_sum1 / (double)frames_measure_num,
						face_cr_mean = inv_sum / (double)frames_measure_num, face_cr_mean1 = inv_sum1 / (double)frames_measure_num,
						bbox_diff_mean = bbox_diff_sum / (double)frames_measure_num, face_diff_mean = face_diff_sum / (double)frames_measure_num,
						bbox_diff_mean_planar = bbox_diff_sum_planar / (double)frames_measure_num;


					cv::imwrite("./images/Camera0_test_" + std::to_string(test_num) + ".jpg", captured_image);
					cv::imwrite("./images/Camera1_test_" + std::to_string(test_num) + ".jpg", captured_image1);
					
					cv::imwrite("./images/Camera0_orig_"+std::to_string(test_num)+ ".jpg", disp_image);
					cv::imwrite("./images/Camera1_orig_" + std::to_string(test_num) + ".jpg", disp_image1);


					if (face_diff_sum < s_threshold * callibInvDiffSum[pyramid_ind] )
					{
						//status_str =  " \n -SPOOFED- ";
						//status_str1 = " \n -SPOOFED- "; 
						status_str =  " \n S ";
						status_str1 = " \n S ";
						result_status = "S";
						status_color = CV_RGB(255,0, 0);

					}
					else if (face_diff_sum > g_threshold * callibInvDiffSum[pyramid_ind])
					{
						//status_str =  " \n -GENUINE- ";
						//status_str1 = " \n -GENUINE- ";
						status_str  = " \n G ";
						status_str1 = " \n G ";
						result_status = "G";
						status_color = CV_RGB(0,255,0);
					}
					else 
					{
						//status_str = " \n -UNDEFINED- ";
						//status_str1 = " \n -UNDEFINED- ";
						status_str =  " \n U ";
						status_str1 = " \n U ";
						result_status = "U";
						status_color = CV_RGB(180, 180, 0);
					}
					std::cout << "\n ____________ results the face status is: " << result_status;

					if (hide_result_annotations) {
						status_str = "";
						status_str1 = "";
					}


					std::cout << " \n face_diff_sum = " << face_diff_sum << "\n callibInvDiffSum = ";
					int ind = 0;
					for (auto value : callibInvDiffSum) {
						if (ind == pyramid_ind) std::cout << " =>";
						std::cout << value << ",";
						ind++;
					}

					ofstream meta_file("./images/metadata_test" + std::to_string(test_num) + ".m");
					if (meta_file.is_open())
					{
						meta_file << "\nPyramidDiff=[";
						for (auto value : callibInvDiffSum)	meta_file << value / (double)frames_measure_num << ",";
						meta_file << "];";

						meta_file << "\nFaceDiff=" << face_diff_sum / (double)frames_measure_num << ";";
						meta_file << "\ns_threshold=" << s_threshold << ";";
						meta_file << "\ng_threshold=" << g_threshold << ";";
						meta_file << "\npyramid_height=" << pyramid_height << ";";
						meta_file << "\npyramid_num=" << pyramid_num << ";";
						meta_file << "\npyramid_ind=" << pyramid_ind << ";";

						meta_file << "\nis_genuine=" << (int)is_genuine_status  << ";";
						meta_file << "\nresult_status= '" << result_status << "';";

						meta_file << "\n angles0=" << euler_angles << ";";
						meta_file << "\n angles1=" << euler_angles1 << ";";

						meta_file << "\n callibrationPnts=" << callibrationPnts[pyramid_ind] << ";";

						meta_file << "\n callibrationPnts1=" << callibrationPnts1[pyramid_ind] << ";";

						meta_file.close();
					}

					if (is_genuine_file.is_open()) {
						is_genuine_file << test_num << " " << (int)is_genuine_status << "\n";
					}


					test_num++;

					if  (outputVideo.isOpened())
							outputVideo.release();
					if (outputVideo1.isOpened())
						outputVideo1.release();


				}else {
					status_str  = "Recording...";
					status_str1 = "Recording...";
					measured_frames++;
				}


				
			}


			char character_press = cv::waitKey(1);

			// restart the trackers
			if (character_press == 'r')
			{
				for (size_t i = 0; i < clnf_models.size(); ++i)
				{
					clnf_models[i].Reset();
					active_models[i] = false;
				}

				for (size_t i = 0; i < clnf_models1.size(); ++i)
				{
					clnf_models1[i].Reset();
					active_models1[i] = false;
				}
			}
			// quit the application
			else if (character_press == 'q')
			{	
				if (is_genuine_file.is_open())
					is_genuine_file.close();
				return(0);
			}
			else if (character_press == 'c')
			{
				status_str = "";
				status_str1 = "";
			}
			else if (character_press == '.')
			{
				selected_landmark_num++;
			}
			else if (character_press == ',') {
				selected_landmark_num--;
			}
			else if (character_press == '\t') {
				device = !device;
			}
			else if (character_press == 'g') {
				is_genuine_status = true;
				std::cout << "\nsessoion of genuine faces\n";
			}
			else if (character_press == 's') {
				is_genuine_status = false;
				std::cout << "\nsession of spoofed faces\n";
			}

			else if (character_press == ' ') {
				if (!measure_invarinats) { //clear previous measurements
					FiveInvariants.clear();
  					FiveInvariants1.clear();
					bboxInvarinats.clear();
					bboxInvarinats1.clear();
					bboxInvarinats_planar.clear();
					bboxInvarinats1_planar.clear();
					callibPntsInvDiff.clear();
					measured_frames = 0;

				}
				measure_invarinats = true;
			}



			// Update the frame count
			frame_count++;
		}
		prev_device = device;


#ifdef RESET_MODEL_PER_SESSION 
		frame_count = 0;
		// Reset the model, for the next video
		for (size_t model = 0; model < clnf_models.size(); ++model)
		{
			clnf_models[model].Reset();
			active_models[model] = false;
		}

		for (size_t model = 0; model < clnf_models1.size(); ++model)
		{
			clnf_models1[model].Reset();
			active_models1[model] = false;
		}
#endif 

	}

	return 0;
}



