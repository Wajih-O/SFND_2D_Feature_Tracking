#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>
#include <string>


struct DataFrame { // represents the available sensor information at the same time instance

    cv::Mat cameraImg; // camera image

    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame

    std::string image_name;
    double extraction_time;
    double description_time;

};


#endif /* dataStructures_h */
