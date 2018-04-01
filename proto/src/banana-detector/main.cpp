#include <iostream>
#include <cstdio>
#include <cstring>

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// https://docs.opencv.org/3.0-beta/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html

#define in_range(val,lo,hi) (lo <= val && val <= hi)
static
void display(const std::vector<cv::Rect> & feats, const cv::Mat & base_image)
{
    cv::Mat image = base_image.clone();
    for ( size_t i = 0; i < feats.size(); i++ )
    {
        //std::cout << feats[i] << std::endl;

        cv::Mat banana_roi = base_image(feats[i]);
        cv::Scalar avg = cv::mean(banana_roi);

        cv::Mat3f avg_mat(cv::Vec3f(avg[0], avg[1], avg[2]));
        cv::Mat3f hsv_mat;

        cv::cvtColor(avg_mat, hsv_mat, cv::COLOR_RGB2HSV);
        cv::Scalar hsv = cv::mean(hsv_mat);

        cv::rectangle(image, feats[i], cv::Scalar(255, 0, 255), 1, 8, 0);
        /*
        if (in_range(avg[2]/avg[1], 1.116 - .2, 1.116 + .2) &&
            in_range(avg[2]/avg[0], 2.172 + .1, 10)) {
            cv::rectangle(image, feats[i], cv::Scalar(0, 255, 0), 1, 8, 0);
            std::cout << feats[i] << std::endl;
            std::cout << hsv << std::endl << std::endl;
        }
        */
        ///*
        #define H_CENTER    195
        #define H_RANGE     5
        #define S_CENTER    0.75
        #define S_RANGE     0.15
        #define V_CENTER    140
        #define V_RANGE     15
        if (in_range(hsv[0], H_CENTER-H_RANGE, H_CENTER+H_RANGE) &&
            in_range(hsv[1], S_CENTER-S_RANGE, S_CENTER+S_RANGE) &&
            in_range(hsv[2], V_CENTER-V_RANGE, V_CENTER+V_RANGE)) {
            cv::rectangle(image, feats[i], cv::Scalar(0, 255, 0), 1, 8, 0);
        }
        //*/
    }
    //cv::resize(image, image, cv::Size(image.rows/10, image.cols/10));
    cv::imshow("banana", image);
}

static
std::vector<cv::Rect> detect(cv::CascadeClassifier & cascade, const cv::Mat & image)
{
    std::vector<cv::Rect> feats;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    // detect them
    cascade.detectMultiScale(gray, feats, 1.1, 1, cv::CASCADE_SCALE_IMAGE, cv::Size(60, 60));

    /*
    for (int i = feats.size() - 1; i >= 0; --i) {

        cv::Mat banana_roi = image(feats[i]);
        cv::Scalar avg = cv::mean(banana_roi);

        cv::Mat3f avg_mat(cv::Vec3f(avg[0], avg[1], avg[2]));
        cv::Mat3f hsv_mat;

        cv::cvtColor(avg_mat, hsv_mat, cv::COLOR_RGB2HSV);
        cv::Scalar hsv = cv::mean(hsv_mat);

        #define H_CENTER    195
        #define H_RANGE     5
        #define S_CENTER    0.75
        #define S_RANGE     0.15
        #define V_CENTER    140
        #define V_RANGE     15
        if (in_range(hsv[0], H_CENTER-H_RANGE, H_CENTER+H_RANGE) &&
            in_range(hsv[1], S_CENTER-S_RANGE, S_CENTER+S_RANGE) &&
            in_range(hsv[2], V_CENTER-V_RANGE, V_CENTER+V_RANGE)) {
        } else {
            feats.pop_back();
        }
    }
    */

    return feats;
}

// banana-detector <image path>
int main(int argc, char * argv[])
{

    if (argc <= 1) {
        std::cerr << "Please provide image" << std::endl;
        std::cerr << "banana-detector <image path>" << std::endl;
        return -1;
    }

    cv::Mat image;
    cv::CascadeClassifier banana_cascade;

    if(!banana_cascade.load("data/haar_banana_cascade.xml")) {
    //if(!banana_cascade.load("data/banana1.xml")) {
        std::cerr << "Error loading face cascade\n" << std::endl;
        return -1;
    };

    const char *  filepath = argv[1];
    int len = strlen(filepath);
    bool still = true;
    if (!strcmp(".mp4", &filepath[len-4])) {
        still = false;
    }

    if (still) {
        image = cv::imread(filepath, cv::IMREAD_COLOR);
        //cv::resize(image, image, cv::Size(300,150));
        std::vector<cv::Rect> bananas = detect(banana_cascade, image);
        display(bananas, image);
        cv::waitKey(0);
    } else {
        cv::VideoCapture cap(filepath);
        while(cap.isOpened()) {
            cap >> image;
            std::vector<cv::Rect> bananas = detect(banana_cascade, image);
            display(bananas, image);
            cv::waitKey(30);
        }
    }
    return 0;
}
