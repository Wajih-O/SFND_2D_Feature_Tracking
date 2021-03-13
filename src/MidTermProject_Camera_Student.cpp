/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>
#include <deque>
#include <unordered_map>
#include <unordered_set>


#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"


void log_keypoints_neighborhood(std::vector<cv::KeyPoint> keypoints, int imgIndex, std::string detectorType) {

    std::cout << "Storing keypoints neighborhood size distribution im: " << imgIndex << ", detector: "<< detectorType << std::endl;
    // Storing keypoints neighborhood size distribution (diameter of the meaningful keypoint neighborhood)
    std::stringstream keypoints_dist_distribution_output_filename;
    keypoints_dist_distribution_output_filename << "keypoints_neighborhood_"<< imgIndex << "_" << detectorType << ".txt";
    std::ofstream keypoints_dist_distribution_output;
    keypoints_dist_distribution_output.open(keypoints_dist_distribution_output_filename.str());
    for (auto keyPoint: keypoints) {
        keypoints_dist_distribution_output << keyPoint.size << std::endl; // stores diameter of the meaningful keypoint neighborhood
    }

}



/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    std::string dataPath = "../";

    // camera
    std::string imgBasePath = dataPath + "images/";
    std::string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    std::string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // detector/descriptor

    std::string detectorType; // (later will take values in)  HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    std::string descriptorType; // (later will take values in) BRIEF, ORB, FREAK, AKAZE, SIFT

    std::string binType = "DES_BINARY";
    std::string hogType = "DES_HOG";

    std::unordered_map <std::string, std::string> descToBinHogType { {"BRIEF",binType}, {"ORB", binType} , {"FREAK", binType}, {"AKAZE", binType}, {"SIFT", hogType}};

    std::string matcherType = "MAT_BF";  // MAT_BF, MAT_FLANN
    // std::string descriptorBinHoGType = "DES_BINARY"; // DES_BINARY, DES_HOG TODO a map for each fo the descriptor previously mentioned
    std::string selectorType = "SEL_KNN"; // SEL_NN, SEL_KNN

    // Experiment output file build
    std::ofstream output_file;
    output_file.open("test.csv");
    std::string separator = ", ";

    // Experiments header
    std::stringstream header;

    header << "detector" << separator;
    header << "descriptor" << separator;
    header << "matcher" << separator;
    header << "selector" << separator;
    header << "im_src" << separator;
    header << "im_ref" << separator;
    header << "im_src_extr_t" << separator;
    header << "im_ref_extr_t" << separator;
    header << "im_src_desc_t" << separator;
    header << "im_ref_desc_t" << separator;
    header << "matching_t" << separator;
    header << "matching_count";


    output_file << header.str() << std::endl;



    // A collection to store the already done keypoint neighborhood
    // as the experiment loop is designed in a way the extraction (is duplicated)
    std::unordered_map<std::string, std::unordered_set<int>> neighborhood_collected;


    std::vector<std::pair<std::string, std::string>> detectorDescriptorCombination {
    {"HARRIS", "BRIEF"}, {"HARRIS", "ORB"}, {"HARRIS", "FREAK"}, {"HARRIS", "SIFT"},
    {"FAST", "BRIEF"}, {"FAST", "ORB"}, {"FAST", "FREAK"}, {"FAST", "SIFT"},
    {"BRISK", "BRIEF"}, {"BRISK", "ORB"}, {"BRISK", "FREAK"}, {"BRISK", "SIFT"},
    {"ORB", "BRIEF"}, {"ORB", "ORB"}, {"ORB", "FREAK"}, {"ORB", "SIFT"},
    {"AKAZE", "BRIEF"}, {"AKAZE", "ORB"}, {"AKAZE", "FREAK"}, {"AKAZE", "SIFT"},
    {"SIFT", "SIFT"}, {"SIFT", "BRIEF"}, {"SIFT", "FREAK"}, // {"SIFT", "ORB"},
    {"AKAZE", "AKAZE"}
    };

    // loop over all compatible detector/descriptor combination

    for (auto item:  detectorDescriptorCombination) {
    std::tie(detectorType, descriptorType) = item;
    std::cout << std::endl <<"- Detector/descriptor comb.:" << detectorType  <<  ", " << descriptorType << std::endl;
    std::cout << std::endl;

    // todo: put back "SIFT" (activating )
    // Before re-activating SIFT
    // terminate called after throwing an instance of 'cv::Exception'
    // what():  OpenCV(4.1.0) /home/wo/tmp1/opencv_contrib-4.1.0/modules/xfeatures2d/src/sift.cpp:1207:
    // error: (-213:The function/feature is not implemented) This algorithm is patented and is excluded in this configuration;
    // Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library in function 'create'

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    std::deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results


    // loop over images
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++) {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
        std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;


        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // TASK MP.1 -> ring buffer impl. of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        frame.image_name = imgFullFilename;

        dataBuffer.push_back(frame);
        if (dataBuffer.size()> dataBufferSize) {
            dataBuffer.pop_front();
        }

        // EOF TASK MP.1

        std::cout << std::endl <<"#1 : image " << imgIndex + 1 << " loaded/reloded into buffer" << std::endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        std::vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        // TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable std::string-based selection based on detectorType
        // -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT


        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, (dataBuffer.end() - 1)->extraction_time, std::cout, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, (dataBuffer.end() - 1)->extraction_time, std::cout, false);
        }
        else if (detectorType.compare("FAST") == 0)
        {
            detKeypointsFAST(keypoints, imgGray, (dataBuffer.end() - 1)->extraction_time, std::cout, false);
        }
        else if (detectorType.compare("BRISK") == 0)
        {
            detKeypointsBRISK(keypoints, imgGray, (dataBuffer.end() - 1)->extraction_time, std::cout, false);
        }
        else if (detectorType.compare("ORB") == 0)
        {
            detKeypointsORB(keypoints, imgGray, (dataBuffer.end() - 1)->extraction_time, std::cout, false);
        }
        else if (detectorType.compare("AKAZE") == 0)
        {
            detKeypointsAKAZE(keypoints, imgGray, (dataBuffer.end() - 1)->extraction_time, std::cout, false);
        }
        else if (detectorType.compare("SIFT") == 0)
        {
            detKeypointsSIFT(keypoints, imgGray, (dataBuffer.end() - 1)->extraction_time, std::cout, false);
        }

        // EOF TASK MP.2


        // TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            auto noFocusKeypointNbr = keypoints.size();
            for (auto iter=keypoints.begin(); iter != keypoints.end();) {
                // test if the keypoint is within the rectangle
                if (vehicleRect.contains((*iter).pt)) {
                    iter++;
                }
                else {
                    iter = keypoints.erase(iter);
                }
            }
        std::cout << "- Focus on vehicule: keypoints filtered " << noFocusKeypointNbr << " -> "<< keypoints.size() << std::endl;
        }
        // EOF TASK MP.3


        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 100;
            if (detectorType.compare("SHITOMASI") == 0) {
                // there is no response info, so keep the first maxKeypoints as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            std::cout << " NOTE: Keypoints have been limited!" << std::endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;


        // Optimized/ detector neighborhood collection with (avoid duplicate logging on detector)
        bool logging_processed = false;
        if (neighborhood_collected.find(detectorType) != neighborhood_collected.end()) {
            // detector found
            auto imagesSet = &neighborhood_collected[detectorType];
            if (imagesSet->find(imgIndex) == imagesSet->end()) {
                log_keypoints_neighborhood((dataBuffer.end() - 1)->keypoints, imgIndex, detectorType);
                imagesSet->insert(imgIndex);
            }
        } else {
            log_keypoints_neighborhood((dataBuffer.end() - 1)->keypoints, imgIndex, detectorType);
            neighborhood_collected[detectorType] = { (int) imgIndex };
        }


        std::cout << "#2 : DETECT KEYPOINTS done" << std::endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        // TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        // -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;

        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, (dataBuffer.end() - 1)->description_time);

        // EOF TASK MP.4

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        std::cout << "#3 : EXTRACT DESCRIPTORS done" << std::endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            std::cout << "Matching last 2 buffer images" << std::endl;

            std::stringstream matching_log;

            // Logging descriptor, detector, matcher and selector
            matching_log << detectorType << separator;
            matching_log << descriptorType << separator;
            matching_log << matcherType << separator;
            matching_log << selectorType << separator;

            matching_log << (dataBuffer.end() - 2)->image_name << separator;
            matching_log << (dataBuffer.end() - 1)->image_name << separator;
            matching_log << (dataBuffer.end() - 2)->extraction_time << separator;
            matching_log << (dataBuffer.end() - 1)->extraction_time << separator;
            matching_log << (dataBuffer.end() - 2)->description_time << separator;
            matching_log << (dataBuffer.end() - 1)->description_time << separator;

            /* MATCH KEYPOINT DESCRIPTORS */

            std::vector<cv::DMatch> matches;
            // TASK MP.5 -> add FLANN matching in file matching2D.cpp (done)
            // TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp (done)
            double matching_time;
            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descToBinHogType[descriptorType], matcherType, selectorType, matching_time);

            matching_log << matching_time << separator;

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            matching_log << (dataBuffer.end() - 1)->kptMatches.size();
            std::cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << std::endl;

            // visualize matches between current and previous image
            bVis = false;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                std::stringstream title_buf;
                title_buf << "Matching keypoints between two camera images ";
                title_buf << " detector " << detectorType;
                cv::namedWindow(title_buf.str(), 7);
                cv::imshow(title_buf.str(), matchImg);
                std::cout << "Press key to continue to next image" << std::endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;

            // logging matching data to the output file
            output_file << matching_log.str() << std::endl;
        }

    } // end of loop over all images

    } // end of loop over all compatible detectordescriptor combination

    return 0;
}
