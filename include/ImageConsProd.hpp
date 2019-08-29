#include <librealsense/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace rs;
using namespace cv;

// 分辨率帧率设置
int const INPUT_WIDTH = 640;
int const INPUT_HEIGHT = 480;
int const DEEPTH_INPUT_WIDTH = 640;
int const DEEPTH_INPUT_HEIGHT = 480;
int const FRAMERATE = 30;

// 窗口显示初始化
char *const WINDOW_DEPTH = "Depth Image";
char *const WINDOW_RGB = "RGB Image";


bool initialize_streaming();
void setup_windows();
bool display_next_frame();
void fake_color(Mat &img);
void result_show();
void image_process(Mat &src);
void depth_process(Mat &src);