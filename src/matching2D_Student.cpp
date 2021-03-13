#include <numeric>
#include <ostream>

#include "matching2D.hpp"

using namespace std;

void visualize_extracted_features(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string windowName){
    // visualize results
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // string windowName = "ORB Detector Results";
    cv::namedWindow(windowName, 6);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType, double& matching_time)
{
    // configure matcher
    bool crossCheck = false;

    std::cout << descSource.size << " <-> " << descRef.size << std::endl;
    if (descSource.type() == CV_32F)
        std::cout << (descSource.type() == CV_32F) << " <-> " << (descRef.type() == CV_32F) << std::endl;
    cv::Ptr<cv::DescriptorMatcher> matcher;


    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;
        if (descSource.type() == CV_32F) {
            normType = cv::NORM_L1;
            }
        else {
            normType = cv::NORM_HAMMING;
        }


        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F) {
            descSource.convertTo(descSource, CV_32F);
        }
        if (descRef.type() != CV_32F) {
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task

    double t = (double)cv::getTickCount();
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in descSource
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> k_matches;
        matcher->knnMatch(descSource, descRef, k_matches, 2);

        // Task 06: match selection and perform descriptor distance ratio filtering with threshold=0.8

        float filtering_threshold = 0.8;
        for (auto item: k_matches) {
            // matches.insert(matches.end(), item.begin(), item.end()); // (just for testing) that would insert all the mathes without filtering
            if (item.size() > 0) {
                if (item.size() >= 2) {
                    // only consider the comparison when the second matching distance is greater than 0
                    // otherwise it is obvious that we are in a very ambiguous matching! and descriptor is skipped
                    if (item[1].distance > 0) {
                        auto distance_ratio = (item[0].distance / item [1].distance); // the fact that the second one has a higher distance
                        if (distance_ratio < filtering_threshold) {
                            matches.push_back(item[0]);
                        }
                    }
                } else {
                    // as it is the only match
                    matches.push_back(item[0]);
                }
            }
            // std::cout << " distance ratio selection " << k_matches.size() << " -> " << matches.size() << std::endl;
        }
    }
    matching_time = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
}


// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double &description_time, std::ostream &log)
{
    // BRIEF, ORB, FREAK, AKAZE, SIFT
    // The descriptor are built with their default parameters
    // TODO: inject parameters using a config (json like)

    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0) {

        // default params (according to the doc): TODO inject params as json config + compatibility check for each of the interest points descriptors
        int bytes = 32;
        bool user_orientation = false;

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0) {

        // to set the WTA_K
        int wta_k = 2;
        extractor = cv::ORB::create();
        // Todo set extractor parameter placeholder (built with the default ones)
        // std::dynamic_pointer_cast<cv::Ptr<cv::ORB>>(extractor).setWTA_K(wta_k);
    }
    else if (descriptorType.compare("FREAK") == 0) {

        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
     else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    description_time = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    log << descriptorType << " descriptor extraction in " << description_time << " ms" << endl;
}


// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &extraction_time, std::ostream &log, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4; //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    extraction_time = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    log << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << extraction_time << " ms" << endl;

    // visualize results
    if (bVis)
    {
        visualize_extracted_features(keypoints, img, "Shi-Tomasi Corner Detector Results");
    }
}


// Detect keypoints in image using Harris detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &extraction_time, std::ostream &log, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    int apertureSize = 3;
    double k = 0.04;
    float cornerness = .004;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k);
    cv::Mat dst_norm = cv::Mat::zeros(img.size(), CV_32FC1);
    normalize(dst, dst_norm, 0, 1, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    for (int i=0; i < dst.rows; i++) {
        for (int j=0; j < dst.cols; j++) {
            if (dst.at<float>(i, j) > cornerness) {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(j, i);
                newKeyPoint.size = blockSize;
                keypoints.push_back(newKeyPoint);
            }
        }
    }

    extraction_time = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    log << "Harris Corner Detector with n=" << keypoints.size() << " keypoints in " << 1000 * extraction_time / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        visualize_extracted_features(keypoints, img, "Harris Corner Detector Results");

    }

}

// Detect keypoints in image using FAST detector
void detKeypointsFAST(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &extraction_time, std::ostream &log, bool bVis)
{
    // compute detector parameters based on image size
    int threshold = 4;
    auto detector = cv::FastFeatureDetector::create(threshold);
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    extraction_time =  1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    log << "FAST detection with n=" << keypoints.size() << " keypoints in " << extraction_time << " ms" << endl;


    // visualize results
    if (bVis) {
         visualize_extracted_features(keypoints, img, "FAST Detector Results");
    }

}

// Detect keypoints in image using BRISK detector
void detKeypointsBRISK(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &extraction_time, std::ostream &log, bool bVis)
{
    // compute detector parameters based on image size
    int threshold = 4;
    int octaves = 3;
    float patternScale = 1.0f;

    auto detector = cv::BRISK::create(threshold, octaves, patternScale);
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    extraction_time = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    log << "BRISK detection with n=" << keypoints.size() << " keypoints in " << extraction_time << " ms" << endl;

    // visualize results
    if (bVis)
    {
        visualize_extracted_features(keypoints, img, "BRISK Detector Results");
    }

}


// Detect keypoints in image using ORB detector
void detKeypointsORB(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &extraction_time, std::ostream &log, bool bVis)
{
    // detector parameters
    int nfeatures = 500;
	float scaleFactor = 1.2f;
	int nlevels = 8;
	int edgeThreshold = 31;
	int  firstLevel = 0;
	int WTA_K = 2;
    cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
	int patchSize = 31;
    int fastThreshold = 20;

    auto detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    extraction_time = 1000 *  ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    log << "ORB detection with n=" << keypoints.size() << " keypoints in " << extraction_time  << " ms" << endl;


    // visualize results
    if (bVis)
    {
        visualize_extracted_features(keypoints, img, "ORB Detector Results");
    }

}


// Detect keypoints in image using AKAZE detector
void detKeypointsAKAZE(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &extraction_time, std::ostream &log, bool bVis)
{
    auto descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
    int descriptor_size = 0;
    int descriptor_channels = 3;
    float threshold = 0.001f;
    int nOctaves = 4;
    int nOctaveLayers = 4;
    auto diffusivity = cv::KAZE::DIFF_PM_G2;

    auto detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    extraction_time = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    log << "AKAZE detection with n=" << keypoints.size() << " keypoints in " << extraction_time  << " ms" << endl;

    // visualize results
    if (bVis)
    {
        visualize_extracted_features(keypoints, img, "AKAZE Detector Results");
    }

}


// Detect keypoints in image using SIFT detector
void detKeypointsSIFT(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &extraction_time, std::ostream &log, bool bVis)
{

    auto detector = cv::xfeatures2d::SiftFeatureDetector::create();
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    extraction_time = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    log << "SIFT detection with n=" << keypoints.size() << " keypoints in " << extraction_time << " ms" << endl;


    // visualize results
    if (bVis)
    {
        visualize_extracted_features(keypoints, img, "SIFT Detector Results");
    }

}