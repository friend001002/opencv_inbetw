#define OPENCV_ENABLE_NONFREE

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <../../sources/contrib/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp>

#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

Mat Concat(Mat& one, Mat& two, Mat& three)
{
  if (   (one.cols   ^ two.cols   ^ three.cols) == 0
      || (one.rows   ^ two.rows   ^ three.rows) == 0
      || (one.type() ^ two.type() ^ three.type()) == 0 )
  {
    throw "Sizes and/or types are different";
  }
  
  int ret_cols { one.cols + two.cols + three.cols };
  
  Mat ret(one.rows, ret_cols, one.type());

  /*for (int r {}; r < one.rows; ++r)
  {
    for (int c {}; c < ret_cols;)
    {
      for (int i {c}; i < one.cols + c; ++i)
      {
        ret.at<Vec3b>(Point(i, r)) = one.at<Vec3b>(Point(i - c, r));

        //cout << "from (" <<  i - c << ", " << r << ") to (" << c << " ," << r << ")\n";
      }

      c += one.cols;

      for (int i {c}; i < two.cols + c; ++i)
      {
        ret.at<Vec3b>(Point(i, r)) = two.at<Vec3b>(Point(i - c, r));

        //cout << "from (" << i - c << ", " << r << ") to (" << c << " ," << r << ")\n";
      }

      c += two.cols;

      for (int i {c}; i < three.cols + c; ++i)
      {
        ret.at<Vec3b>(Point(i, r)) = three.at<Vec3b>(Point(i - c, r));

        //cout << "from (" << i - c << ", " << r << ") to (" << c << " ," << r << ")\n";
      }

      c += three.cols;
    }
  }*/

  hconcat(one, two, ret);
  hconcat(ret, three, ret);

  return ret;
}

Mat Diff(Mat& one, Mat& two)
{
  if (   one.cols   != two.cols 
      || one.rows   != two.rows 
      || one.type() != two.type())
  {
    throw "Sizes and/or types are different";
  }

  Mat ret(one.rows, one.cols, one.type());

  for (int r {}; r < one.rows; ++r)
  {
    for (int c {}; c < one.cols; ++c)
    {
      if (   one.at<Vec3b>(Point(c, r))[0] != two.at<Vec3b>(Point(c, r))[0] 
          || one.at<Vec3b>(Point(c, r))[1] != two.at<Vec3b>(Point(c, r))[1] 
          || one.at<Vec3b>(Point(c, r))[2] != two.at<Vec3b>(Point(c, r))[2])
      {
        ret.at<Vec3b>(Point(c, r)) = Vec3b(255, 0, 0);
      }
    }
  }

  return ret;
}

int main(int argc, char** argv)
{
  Mat img_1, img_2, img_inb, img_res;

  img_1 = imread("pic_1.png", IMREAD_COLOR);
  img_2 = imread("pic_2.png", IMREAD_COLOR);

  img_inb = Mat(img_1.rows, img_1.cols, img_1.type());
  img_inb.setTo(Scalar(255, 255, 255));

  int minHessian { 400 };
  Ptr<SURF> detector = SURF::create(minHessian);

  if (nullptr == detector)
  {
    return -1;
  }

  vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descs_1, descs_2;

  detector->detectAndCompute(img_1, noArray(), keypoints_1, descs_1);
  detector->detectAndCompute(img_2, noArray(), keypoints_2, descs_2);
  
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

  if (nullptr == matcher)
  {
    return -1;
  }

  vector<vector<DMatch>> knn_matches;

  matcher->knnMatch(descs_1, descs_2, knn_matches, 2);
  
  const float ratio_thresh { 0.7F };
  vector<DMatch> good_matches;

  for (size_t i {}; i < knn_matches.size(); i++)
  {
    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  
  drawMatches(  img_1, keypoints_1, img_2, keypoints_2, good_matches, img_inb 
              , Scalar::all(-1), Scalar::all(-1), vector<char>() 
              , DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  img_res = Concat(img_1, img_inb, img_2);

  if (img_1.empty() || img_2.empty())
  {
    cout << "Could not open or find the images" << std::endl;
    return -1;
  }

  /*namedWindow("Window 1", WINDOW_AUTOSIZE);
  imshow("Window 1", img_1);

  namedWindow("Window 2", WINDOW_AUTOSIZE);
  imshow("Window 2", img_2);*/

  namedWindow("Window inb", WINDOW_AUTOSIZE);
  imshow("Window inb", img_inb);

  //namedWindow("Window res", WINDOW_KEEPRATIO);
  //imshow("Window res", img_res);

  waitKey(0);

  return 0;
}
