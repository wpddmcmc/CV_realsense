#include "ImageConsProd.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

context _rs_ctx;							 //设备
device &_rs_camera = *_rs_ctx.get_device(0); //开始流传输的实感设备
intrinsics _depth_intrin;					 //包含当前深度帧信息的 LRS 内联对象
intrinsics _color_intrin;					 //包含当前色帧信息的 LRS 内联对象
bool _loop = true;							 //用于知道何时停止图像处理

/////////////////////////////////////////////////////////////////////////////
//初始化数据流
/////////////////////////////////////////////////////////////////////////////
bool initialize_streaming()
{
	bool success = false;
	if (_rs_ctx.get_device_count() > 0)
	{
		_rs_camera.enable_stream(rs::stream::color, INPUT_WIDTH, INPUT_HEIGHT, rs::format::bgr8, FRAMERATE);
		_rs_camera.enable_stream(rs::stream::depth, DEEPTH_INPUT_WIDTH, DEEPTH_INPUT_HEIGHT, rs::format::z16, FRAMERATE);
		_rs_camera.start();

		success = true;
	}
	return success;
}

/////////////////////////////////////////////////////////////////////////////
//初始化窗口
/////////////////////////////////////////////////////////////////////////////
void setup_windows()
{
	namedWindow(WINDOW_DEPTH, 0);
	namedWindow(WINDOW_RGB, 0);
	resizeWindow(WINDOW_DEPTH, 640, 480);
	resizeWindow(WINDOW_RGB, 640, 480);
}

/////////////////////////////////////////////////////////////////////////////
// 在 OpenCV 窗口中显示 LRS 数据
/////////////////////////////////////////////////////////////////////////////
bool display_next_frame()
{
	_rs_camera.wait_for_frames();

	uchar * color_frame = (uchar *)(_rs_camera.get_frame_data(rs::stream::color));
	uint16_t * depth_frame = (uint16_t *)(_rs_camera.get_frame_data(rs::stream::depth));

	Mat rgb(480,
			640,
			CV_8UC3,
			color_frame);

	Mat depth16(480,
				640,
				CV_16UC1,
				depth_frame);
	Mat depth8u = depth16;
	depth8u.convertTo(depth8u, CV_8UC1, 255.0 / 1000);	
	imshow(WINDOW_RGB, rgb);
    imshow(WINDOW_DEPTH,depth8u);
    Mat image_src;
	rgb.copyTo(image_src);
	image_process(image_src);
	imshow("2",image_src);
	fake_color(depth8u);
	depth_process(depth8u);
	return true;
}

/////////////////////////////////////////////////////////////////////////////
// 伪彩虹
/////////////////////////////////////////////////////////////////////////////
void fake_color( Mat &img)	
{
	Mat img_color(480,640,CV_8UC3);
	uchar tmp=0;
	for(int x=0;x<img.rows;x++)
	{
		for(int y=0;y<img.cols;y++)
		{
			tmp = img.at<uchar>(x,y);
			if(tmp<=51)
			{
				img_color.at<Vec3b>(x,y)[0] = 255;
				img_color.at<Vec3b>(x,y)[1] = tmp*5;
				img_color.at<Vec3b>(x,y)[2] = 0;
			}
			else if(tmp<=102)
			{
				tmp-=51;
				img_color.at<Vec3b>(x,y)[0] = 255-tmp*5;
				img_color.at<Vec3b>(x,y)[1] = 255;
				img_color.at<Vec3b>(x,y)[2] = 0;
			}
			else if(tmp<=153)
			{
				tmp-=153;
				img_color.at<Vec3b>(x,y)[0] = 0;
				img_color.at<Vec3b>(x,y)[1] = tmp*2.5;//255-uchar(128.0*tmp/51.0+0.5);;
				img_color.at<Vec3b>(x,y)[2] = 255;
			}
			else if(tmp<=204)
			{
				tmp-=153;
				img_color.at<Vec3b>(x,y)[0] = 0;
				img_color.at<Vec3b>(x,y)[1] = tmp*2.5;//255-uchar(128.0*tmp/51.0+0.5);
				img_color.at<Vec3b>(x,y)[2] = 255;
			}
			else{
				tmp-=153;
				img_color.at<Vec3b>(x,y)[0] = 0;
				img_color.at<Vec3b>(x,y)[1] = tmp*2.5;//127-uchar(128.0*tmp/51.0+0.5);
				img_color.at<Vec3b>(x,y)[2] = 255;
			}
		}
	}
	blur( img_color, img_color, Size(5,5) );
	morphologyEx(img_color, img_color, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(5, 5)));
	imshow("img_color",img_color);
	img_color.copyTo(img);
}

void result_show()
{
	rs::log_to_console(rs::log_severity::warn);
	if (!initialize_streaming())
	{
		std::cout << "Unable to locate a camera" << std::endl;
		rs::log_to_console(rs::log_severity::fatal);
	}
    setup_windows();	
    while(true){
        int key;
        display_next_frame();
        key = waitKey(30);
        if ((char)key == 'q') break;
    }
	_rs_camera.stop();
	destroyAllWindows();
}

void image_process(Mat &src)
{
	Mat gray;
	src.copyTo(gray);
	blur( gray, gray, Size(3,3) );
	cvtColor(gray,gray,COLOR_BGR2GRAY);
	threshold(gray, gray, 80, 255, THRESH_BINARY); //brightness threshold
	morphologyEx(gray, gray, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(5, 5))); //dilate to get more obvious contour
	Canny( gray, gray, 3, 9,3 );
    vector< vector<Point2i> > contours;
	vector<Vec4i> hierarchy;
	findContours(gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	drawContours(src,contours,-1,Scalar(255,0,0),3,8);
	imshow("Open",gray);

	vector<Rect> boundRect(contours.size());
	vector<Rect> numberRect;
	vector<vector<Point>> contours_poly(contours.size());
	Mat number_target;
	for (int a = 0; a < contours.size(); a++)
	{
		// 进行多边形拟合
		approxPolyDP(Mat(contours[a]), contours_poly[a], 3, true);
		int s = contours_poly[a].size();
		// 最小包围矩形
		boundRect[a] = boundingRect(Mat(contours_poly[a]));
		double rectsize = contourArea(contours_poly[a]);
		if (rectsize > 150&& rectsize <8000)						//筛选面积
			numberRect.push_back(boundRect[a]); //得到数字包围矩形ROI
		}
		 
}

void depth_process(Mat &src)
{
	Mat gray;
	blur( src, gray, Size(3,3) );
	cvtColor(gray,gray,COLOR_BGR2GRAY);
	threshold(gray, gray, 80, 255, THRESH_BINARY); //brightness threshold
	imshow("gray",gray);
}