# Face anti-spoofing based on projective invariants

An implementation Geoemetric Face anti-spoofing algorithm based on the paper
```
Naitsat, Alexander, and Yehoshua Y. Zeevi. "Face anti-spoofing based on projective invariants." 
2018 IEEE International Conference on the Science of Electrical Engineering in Israel (ICSEE). IEEE, 2018.
```


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) (face landmark detection)
* [opencv](https://opencv.org/) (OpenFace dependency)
* [dlib](http://dlib.net/) (OpenFace dependency)

### Installing

To run the algorithm clone OpenFace into OpenFace-master and replace the following files by files from the repository root 

* FaceLandmarkVidMulti.cpp  -> OpenFace-master\exe\FaceLandmarkVidMulti\FaceLandmarkVidMulti.cpp 
* LandmarkDetectorUtils.h   -> OpenFace-master\lib\local\LandmarkDetector\include\LandmarkDetectorUtils.h
* LandmarkDetectorUtils.cpp -> OpenFace-master\lib\local\LandmarkDetector\src\LandmarkDetectorUtils.cpp

### Complilation 
* Compile OpenFace with the replaced files.
* Use Visual Studio solution fiel on windows platforms. (Tested with Visual Studio 2015). 
* On other platforms use OpenFace cmake file 


### Running the Demo 

1.  Connect two RGB cameras and run OpenFace-master\x64\Release\FaceLandmarkVidMulti.exe
2.  Use the following keys to control the demo
*  Space bar to start recording. After the recording is finished results are displayed and saved in file according to given flags
*  Press 'g' and 's' to label recorded videos as  "genuine" and "spoofed" videos, respectively.
3. Use the following command line flags to control the algorithm 
* '-record_video' to save recording of each camera in a separate avi file in   exe\FaceLandmarkVidMulti\videos
* '-load_video' to load stereo video stream  from avi files listed in \exe\FaceLandmarkVidMulti\video_names_camera0.txt, video_names_camera1.txt
* '-annotation' to display classification results after each recording 
*  '-show_angle' to display  Euler angles  of the head pose.


### Visualization and analysis 
