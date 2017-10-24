// ------------------------- OpenPose Library Tutorial - Thread - Example 2 - Synchronous -------------------------
// Synchronous mode: ideal for performance. The user can add his own frames producer / post-processor / consumer to the OpenPose wrapper or use the default ones.

// This example shows the user how to use the OpenPose wrapper class:
    // 1. Extract and render keypoint / heatmap / PAF of that image
    // 2. Save the results on disc
    // 3. Display the rendered pose
    // Everything in a multi-thread scenario
// In addition to the previous OpenPose modules, we also need to use:
    // 1. `core` module:
        // For the Array<float> class that the `pose` module needs
        // For the Datum struct that the `thread` module sends between the queues
    // 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively
// This file should only be used for the user to take specific examples.



// C++ std library dependencies
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <string>
#include <thread> // std::this_thread
#include <vector>
// Other 3rdparty dependencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging

// OpenPose dependencies
// Option a) Importing all modules
#include <openpose/headers.hpp>
// Option b) Manually importing the desired modules. Recommended if you only intend to use a few modules.
// #include <openpose/core/headers.hpp>
// #include <openpose/experimental/headers.hpp>
// #include <openpose/face/headers.hpp>
// #include <openpose/filestream/headers.hpp>
// #include <openpose/gui/headers.hpp>
// #include <openpose/pose/headers.hpp>
// #include <openpose/producer/headers.hpp>
// #include <openpose/thread/headers.hpp>
// #include <openpose/utilities/headers.hpp>
// #include <openpose/wrapper/headers.hpp>

// See all the available parameter options withe the `--help` flag. E.g. `./build/examples/openpose/openpose.bin --help`.
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_dir,                "",      "Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).");
// Producer
DEFINE_int32(camera, -1, "The camera index for cv::VideoCapture. Integer in the range [0, 9]. Select a negative"
	" number (by default), to auto-detect and open the first available camera.");
DEFINE_string(camera_resolution, "1280x720", "Size of the camera frames to ask for.");
DEFINE_double(camera_fps, 30.0, "Frame rate for the webcam (only used when saving video from webcam). Set this value to the"
	" minimum value between the OpenPose displayed speed and the webcam real frame rate.");
DEFINE_string(video, "", "Use a video file instead of the camera. Use `examples/media/video.avi` for our default"
	" example video.");
DEFINE_uint64(frame_first, 0, "Start on desired frame number. Indexes are 0-based, i.e. the first frame has index 0.");
DEFINE_uint64(frame_last, -1, "Finish on desired frame number. Select -1 to disable. Indexes are 0-based, e.g. if set to"
	" 10, it will process 11 frames (0-10).");
DEFINE_bool(frame_flip, false, "Flip/mirror each frame (e.g. for real time webcam demonstrations).");
DEFINE_int32(frame_rotate, 0, "Rotate each frame, 4 possible values: 0, 90, 180, 270.");
DEFINE_bool(frames_repeat, false, "Repeat frames when finished.");
// OpenPose
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(resolution,               "1280x720",     "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " default images resolution.");
DEFINE_int32(num_gpu,                   -1,             "The number of GPU devices to use. If negative, it will use all the available GPUs in your"
                                                        " machine.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_int32(keypoint_scale,            0,              "Scaling of the (x,y) coordinates of the final pose data array, i.e. the scale of the (x,y)"
                                                        " coordinates that will be saved with the `write_keypoint` & `write_keypoint_json` flags."
                                                        " Select `0` to scale it to the original source resolution, `1`to scale it to the net output"
                                                        " size (set with `net_resolution`), `2` to scale it to the final output size (set with"
                                                        " `resolution`), `3` to scale it in the range [0,1], and 4 for range [-1,1]. Non related"
                                                        " with `scale_number` and `scale_gap`.");
// OpenPose Body Pose
DEFINE_string(model_pose,               "COCO",         "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                                        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased,"
                                                        " the speed increases. For maximum speed-accuracy balance, it should keep the closest aspect"
                                                        " ratio possible to the images or videos to be processed. E.g. the default `656x368` is"
                                                        " optimal for 16:9 videos, e.g. full HD (1980x1080) and HD (1280x720) videos.");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                                                        " If you want to change the initial scale, you actually want to multiply the"
                                                        " `net_resolution` by your desired initial scale.");
DEFINE_bool(heatmaps_add_parts,         false,          "If true, it will add the body part heatmaps to the final op::Datum::poseHeatMaps array"
                                                        " (program speed will decrease). Not required for our library, enable it only if you intend"
                                                        " to process this information later. If more than one `add_heatmaps_X` flag is enabled, it"
                                                        " will place then in sequential memory order: body parts + bkg + PAFs. It will follow the"
                                                        " order on POSE_BODY_PART_MAPPING in `include/openpose/pose/poseParameters.hpp`.");
DEFINE_bool(heatmaps_add_bkg,           false,          "Same functionality as `add_heatmaps_parts`, but adding the heatmap corresponding to"
                                                        " background.");
DEFINE_bool(heatmaps_add_PAFs,          false,          "Same functionality as `add_heatmaps_parts`, but adding the PAFs.");
DEFINE_int32(heatmaps_scale,            2,              "Set 0 to scale op::Datum::poseHeatMaps in the range [0,1], 1 for [-1,1]; and 2 for integer"
                                                        " rounded [0,255].");
// OpenPose Face
DEFINE_bool(face,                       false,          "Enables face keypoint detection. It will share some parameters from the body pose, e.g."
                                                        " `model_folder`. Note that this will considerable slow down the performance and increse"
                                                        " the required GPU memory. In addition, the greater number of people on the image, the"
                                                        " slower OpenPose will be.");
DEFINE_string(face_net_resolution,      "368x368",      "Multiples of 16 and squared. Analogous to `net_resolution` but applied to the face keypoint"
                                                        " detector. 320x320 usually works fine while giving a substantial speed up when multiple"
                                                        " faces on the image.");
// OpenPose Hand
DEFINE_bool(hand,                       false,          "Enables hand keypoint detection. It will share some parameters from the body pose, e.g."
                                                        " `model_folder`. Analogously to `--face`, it will also slow down the performance, increase"
                                                        " the required GPU memory and its speed depends on the number of people.");
DEFINE_string(hand_net_resolution,      "368x368",      "Multiples of 16 and squared. Analogous to `net_resolution` but applied to the hand keypoint"
                                                        " detector.");
DEFINE_int32(hand_scale_number,         1,              "Analogous to `scale_number` but applied to the hand keypoint detector. Our best results"
                                                        " were found with `hand_scale_number` = 6 and `hand_scale_range` = 0.4");
DEFINE_double(hand_scale_range,         0.4,            "Analogous purpose than `scale_gap` but applied to the hand keypoint detector. Total range"
                                                        " between smallest and biggest scale. The scales will be centered in ratio 1. E.g. if"
                                                        " scaleRange = 0.4 and scalesNumber = 2, then there will be 2 scales, 0.8 and 1.2.");

DEFINE_bool(hand_tracking,              false,          "Adding hand tracking might improve hand keypoints detection for webcam (if the frame rate"
                                                        " is high enough, i.e. >7 FPS per GPU) and video. This is not person ID tracking, it"
                                                        " simply looks for hands in positions at which hands were located in previous frames, but"
                                                        " it does not guarantee the same person ID among frames");
// OpenPose Rendering
DEFINE_int32(part_to_show,              0,              "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body"
                                                        " part heat map, 19 for the background heat map, 20 for all the body part heat maps"
                                                        " together, 21 for all the PAFs, 22-40 for each body part pair PAF");
DEFINE_bool(disable_blending,           false,          "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
                                                        " background, instead of being rendered into the original image. Related: `part_to_show`,"
                                                        " `alpha_pose`, and `alpha_pose`.");
// OpenPose Rendering Pose
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_int32(render_pose,               2,              "Set to 0 for no rendering, 1 for CPU rendering (slightly faster), and 2 for GPU rendering"
                                                        " (slower but greater functionality, e.g. `alpha_X` flags). If rendering is enabled, it will"
                                                        " render both `outputData` and `cvOutputData` with the original image and desired body part"
                                                        " to be shown (i.e. keypoints, heat maps or PAFs).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
                                                        " heatmap, 0 will only show the frame. Only valid for GPU rendering.");
// OpenPose Rendering Face
DEFINE_double(face_render_threshold,    0.4,            "Analogous to `render_threshold`, but applied to the face keypoints.");
DEFINE_int32(face_render,               -1,             "Analogous to `render_pose` but applied to the face. Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_double(face_alpha_pose,          0.6,            "Analogous to `alpha_pose` but applied to face.");
DEFINE_double(face_alpha_heatmap,       0.7,            "Analogous to `alpha_heatmap` but applied to face.");
// OpenPose Rendering Hand
DEFINE_double(hand_render_threshold,    0.2,            "Analogous to `render_threshold`, but applied to the hand keypoints.");
DEFINE_int32(hand_render,               -1,             "Analogous to `render_pose` but applied to the hand. Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_double(hand_alpha_pose,          0.6,            "Analogous to `alpha_pose` but applied to hand.");
DEFINE_double(hand_alpha_heatmap,       0.7,            "Analogous to `alpha_heatmap` but applied to hand.");
// Result Saving
DEFINE_string(write_images,             "",             "Directory to write rendered frames in `write_images_format` image format.");
DEFINE_string(write_images_format,      "png",          "File extension and format for `write_images`, e.g. png, jpg or bmp. Check the OpenCV"
                                                        " function cv::imwrite for all compatible extensions.");
DEFINE_string(write_video,              "",             "Full file path to write rendered frames in motion JPEG video format. It might fail if the"
                                                        " final path does not finish in `.avi`. It internally uses cv::VideoWriter.");
DEFINE_string(write_keypoint,           "",             "Directory to write the people body pose keypoint data. Set format with `write_keypoint_format`.");
DEFINE_string(write_keypoint_format,    "yml",          "File extension and format for `write_keypoint`: json, xml, yaml & yml. Json not available"
                                                        " for OpenCV < 3.0, use `write_keypoint_json` instead.");
DEFINE_string(write_keypoint_json,      "",             "Directory to write people pose data in *.json format, compatible with any OpenCV version.");
DEFINE_string(write_coco_json,          "",             "Full file path to write people pose data with *.json COCO validation format.");
DEFINE_string(write_heatmaps,           "",             "Directory to write heatmaps in *.png format. At least 1 `add_heatmaps_X` flag must be"
                                                        " enabled.");
DEFINE_string(write_heatmaps_format,    "png",          "File extension and format for `write_heatmaps`, analogous to `write_images_format`."
                                                        " Recommended `png` or any compressed and lossless format.");

// Display
DEFINE_bool(fullscreen, false, "Run in full-screen mode (press f during runtime to toggle).");
DEFINE_bool(process_real_time, true, "Enable to keep the original source frame rate (e.g. for video). If the processing time is"
	" too long, it will skip frames. If it is too fast, it will slow it down.");
DEFINE_bool(no_gui_verbose, true, "Do not write text on output images on GUI (e.g. number of current frame and people). It"
	" does not affect the pose rendering.");
DEFINE_bool(no_display, false, "Do not open a display window. Useful if there is no X server and/or to slightly speed up"
	" the processing if visual output is not required.");

//windows
#define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#include <WINSOCK2.H>
#pragma comment(lib,"ws2_32.lib")
// windows
int startConnect()
{
	int err;
	WORD versionRequired;
	WSADATA wsaData;
	versionRequired = MAKEWORD(1, 1);
	err = WSAStartup(versionRequired, &wsaData);//协议库的版本信息
	if (!err)
	{
		printf("客户端嵌套字已经打开!\n");
	}
	else
	{
		printf("客户端的嵌套字打开失败!\n");
		return 0;//结束
	}
	SOCKET client = socket(AF_INET, SOCK_STREAM, 0);
	SOCKADDR_IN clientsock_in;
	clientsock_in.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	clientsock_in.sin_family = AF_INET;
	clientsock_in.sin_port = htons(8001);

	connect(client, (SOCKADDR*)&clientsock_in, sizeof(SOCKADDR));//开始连接
	char receiveBuf[100];
	recv(client, receiveBuf, 101, 0);
	printf("%s\n", receiveBuf);
	send(client, "ttt", strlen("ttt") + 1, 0);
	return client;
}

SOCKET clientSocket = startConnect();


int stopConnect() {
	closesocket(clientSocket);
	WSACleanup();
	return 1;
}

int sendNotify() {
	send(clientSocket, "nnn", strlen("nnn") + 1, 0);
	return 1;
}


// If the user needs his own variables, he can inherit the op::Datum struct and add them
// UserDatum can be directly used by the OpenPose wrapper because it inherits from op::Datum, just define Wrapper<UserDatum> instead of
// Wrapper<op::Datum>
struct UserDatum : public op::Datum
{
    bool boolThatUserNeedsForSomeReason;

    UserDatum(const bool boolThatUserNeedsForSomeReason_ = false) :
        boolThatUserNeedsForSomeReason{boolThatUserNeedsForSomeReason_}
    {}
};

// The W-classes can be implemented either as a template or as simple classes given
// that the user usually knows which kind of data he will move between the queues,
// in this case we assume a std::shared_ptr of a std::vector of UserDatum

// This worker will just read and return all the jpg files in a directory
class WUserInput : public op::WorkerProducer<std::shared_ptr<std::vector<UserDatum>>>
{
public:
    WUserInput(const std::string& directoryPath) :
        //mImageFiles{op::getFilesOnDirectory(directoryPath)},
        mImageFiles{op::getFilesOnDirectory(directoryPath, std::vector<std::string>{"jpg", "png"})}, // If we want "jpg" + "png" images
        mCounter{0}
    {
        if (mImageFiles.empty())
            op::error("No images found on: " + directoryPath, __LINE__, __FUNCTION__, __FILE__);
    }

    void initializationOnThread() {}

    std::shared_ptr<std::vector<UserDatum>> workProducer()
    {
        try
        {
            // Close program when empty frame
            if (mImageFiles.size() <= mCounter)
            {
                op::log("Last frame read and added to queue. Closing program after it is processed.", op::Priority::High);
                // This funtion stops this worker, which will eventually stop the whole thread system once all the frames have been processed
                this->stop();
                return nullptr;
            }
            else
            {
                // Create new datum
                auto datumsPtr = std::make_shared<std::vector<UserDatum>>();
                datumsPtr->emplace_back();
                auto& datum = datumsPtr->at(0);

                // Fill datum
				auto& fullPath = mImageFiles.at(mCounter++);
                datum.cvInputData = cv::imread(fullPath);
				datum.name = op::getFileNameNoExtension(fullPath);

                // If empty frame -> return nullptr
                if (datum.cvInputData.empty())
                {
                    op::log("Empty frame detected on path: " + mImageFiles.at(mCounter-1) + ". Closing program.", op::Priority::High);
                    this->stop();
                    datumsPtr = nullptr;
                }

                return datumsPtr;
            }
        }
        catch (const std::exception& e)
        {
            op::log("Some kind of unexpected error happened.");
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

private:
    const std::vector<std::string> mImageFiles;
    unsigned long long mCounter;
};

// This worker will just invert the image
class WUserPostProcessing : public op::Worker<std::shared_ptr<std::vector<UserDatum>>>
{
public:
    WUserPostProcessing()
    {
        // User's constructor here
    }

    void initializationOnThread() {}

    void work(std::shared_ptr<std::vector<UserDatum>>& datumsPtr)
    {
        // User's post-processing (after OpenPose processing & before OpenPose outputs) here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKeypoints: Array<float> with the estimated pose
        try
        {
			if (datumsPtr != nullptr && !datumsPtr->empty()) {
				for (auto& datum : *datumsPtr) {

					//op::log("Face keypoints: " + datum.faceKeypoints.toString());
					//op::log("Left hand keypoints: " + datum.handKeypoints[0].toString());
					//op::log("Right hand keypoints: " + datum.handKeypoints[1].toString());

					cv::Mat &img = datum.cvOutputData;
					const auto& handRects = datum.handRectangles;

					auto left_points = datum.handKeypoints.at(0);
					bool bare_hand = handConfirm(left_points, datum, 0);

					auto right_points = datum.handKeypoints.at(1);	
					bare_hand = handConfirm(right_points, datum, 1) || bare_hand;

					if (bare_hand) {
						/*
						cv::String txt = "Bare hand!";
						int font_face = cv::FONT_HERSHEY_COMPLEX;
						double font_scale = 1;
						int thickness = 2;
						int baseline = 0;
						cv::Scalar color(0, 255, 255);
						cv::Size size = cv::getTextSize(txt, font_face, font_scale, thickness, &baseline);
						cv::putText(img, txt, cv::Point(5, 5 + size.height), font_face, font_scale, color, thickness);
						*/
						std::string filename = "F:\\genee_work\\detect-bare-hand\\image\\current.png";
						cv::imwrite(filename, datum.cvInputData);
						sendNotify();
					}
				}
			}
        }
        catch (const std::exception& e)
        {
            op::log("Some kind of unexpected error happened.");
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

	bool handConfirm(const op::Array<float>& handKeypoints, UserDatum& datum, int leftright)
	{

		if (handKeypoints.getVolume() == 0) {
			return false;
		}

		cv::Mat &img = datum.cvOutputData;
		cv::Mat &input_img = datum.cvInputData;
		const auto& handRects = datum.handRectangles;

		auto sizes = handKeypoints.getSize();

		int count = sizes.at(0);
		int row_count = sizes.at(1);
		int point_count = sizes.at(2);

		bool bare_hand = false;

		for (int person = 0; person < count; person++) {
			int not_detected = 0;
			int base_index = person*row_count*point_count;
			for (int row = 0; row < row_count; row++) {
				int index = base_index + row*point_count + 2;
				if (handKeypoints[index] < 0.2) {
					not_detected++;
				}
			}
			if (not_detected < 5) {
				if (isBareHand(img, handKeypoints, person, datum.scaleInputToOutput)) {
					auto rc = handRects.at(person).at(leftright);
					drawHandRect(img, rc, datum.scaleInputToOutput);
					drawHandRect(datum.cvInputData, rc, 1.0);
					bare_hand = true;
				}
			}
		}
		return bare_hand;
	}

	bool isBareHand(cv::Mat& img, const op::Array<float>& handKeypoints, int person, float scaleInputToOutput)
	{
		auto sizes = handKeypoints.getSize();
		int count = sizes.at(0);
		int row_count = sizes.at(1);
		int point_count = sizes.at(2);

		int not_naked = 0;
		int base_index = person*row_count*point_count;
		for (int row = 0; row < row_count; row++) {
			int index = base_index + row*point_count;
			float x = handKeypoints[index] * scaleInputToOutput;
			float y = handKeypoints[index + 1] * scaleInputToOutput;
			cv::Point p(x, y);
			
			cv::Vec3b p3b = img.at<cv::Vec3b>(p);
			int b = p3b[0];
			int g = p3b[1];
			int r = p3b[2];
			float H = getH(r, g, b);

			op::log("H:"+std::to_string(H));

			if (340 > H && H > 40){//TODO
				not_naked++;
			}
			if (not_naked > 5) {
				return false;
			}
		}
		return true;
	}

	float getH(float r, float g, float b) {
		float max = r;
		if (g > max) max = g;
		if (b > max) max = b;
		float min = r;
		if (g < min) min = g;
		if (b < min) min = b;
		float H;
		if (max == r) {
			H = (60 * (g - b)) / (max - min);
		}
		else if (max == g) {
			H = 120 + ((60 * (b - r)) / (max - min));
		}
		else {
			H = 240 + ((60 * (r - g)) / (max - min));
		}
		if (H < 0) H += 360.0;
		return H;
	}

	void drawHandRect(cv::Mat& img, const op::Rectangle<float>& rc, float scaleInputToOutput)
	{
		//从实际来看，缩减1/3还是比较准
		cv::Rect rect(rc.x + rc.width / 6, rc.y + rc.height / 6, rc.width * 2 / 3, rc.height * 2 / 3);

		rect.x *= scaleInputToOutput;
		rect.y *= scaleInputToOutput;
		rect.width *= scaleInputToOutput;
		rect.height *= scaleInputToOutput;

		cv::Rect rectImg(0, 0, img.cols, img.rows);
		rect &= rectImg;//排除图像外区域
		if (rect.x >= 0 && rect.y >= 0 && rect.width > 0 && rect.height > 0) {
			cv::Scalar color(0, 255, 255);
			cv::rectangle(img, rect, color, 2);
		}
	}
};

// This worker will just read and return all the jpg files in a directory
class WUserOutput : public op::WorkerConsumer<std::shared_ptr<std::vector<UserDatum>>>
{
public:
    void initializationOnThread() {}

    void workConsumer(const std::shared_ptr<std::vector<UserDatum>>& datumsPtr)
    {
        try
        {
            // User's displaying/saving/other processing here
                // datum.cvOutputData: rendered frame with pose or heatmaps
                // datum.poseKeypoints: Array<float> with the estimated pose
            if (datumsPtr != nullptr && !datumsPtr->empty())
            {
                // Show in command line the resulting pose keypoints for body, face and hands
                //op::log("\nKeypoints:");
                // Accesing each element of the keypoints
                const auto& poseKeypoints = datumsPtr->at(0).poseKeypoints;
				cv::imshow("User worker GUI", datumsPtr->at(0).cvOutputData);
				cv::waitKey(1);

            }
        }
        catch (const std::exception& e)
        {
            op::log("Some kind of unexpected error happened.");
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
};



int openPoseTutorialWrapper2()
{
    // logging_level
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    // op::ConfigureLog::setPriorityThreshold(op::Priority::None); // To print all logging messages

    op::log("Starting pose estimation demo.", op::Priority::High);
    const auto timerBegin = std::chrono::high_resolution_clock::now();

    // Applying user defined configuration - Google flags to program variables
    // outputSize
    const auto outputSize = op::flagsToPoint(FLAGS_resolution, "1280x720");
    // netInputSize
    const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "656x368");
    // faceNetInputSize
    const auto faceNetInputSize = op::flagsToPoint(FLAGS_face_net_resolution, "368x368 (multiples of 16)");
    // handNetInputSize
    const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");

	// producerType
	const auto producerSharedPtr = op::flagsToProducer(FLAGS_image_dir, FLAGS_video, FLAGS_camera, FLAGS_camera_resolution, FLAGS_camera_fps);


    // poseModel
    const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
    // keypointScale
    const auto keypointScale = op::flagsToScaleMode(FLAGS_keypoint_scale);
    // heatmaps to add
    const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg, FLAGS_heatmaps_add_PAFs);
    op::check(FLAGS_heatmaps_scale >= 0 && FLAGS_heatmaps_scale <= 2, "Non valid `heatmaps_scale`.", __LINE__, __FUNCTION__, __FILE__);
    const auto heatMapScale = (FLAGS_heatmaps_scale == 0 ? op::ScaleMode::PlusMinusOne
                               : (FLAGS_heatmaps_scale == 1 ? op::ScaleMode::ZeroToOne : op::ScaleMode::UnsignedChar ));
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);

    // Initializing the user custom classes

    // Processing
    auto wUserPostProcessing = std::make_shared<WUserPostProcessing>();
    // GUI (Display)
  //  auto wUserOutput = std::make_shared<WUserOutput>();

    op::Wrapper<std::vector<UserDatum>> opWrapper;
	/*
    // Add custom input
	// Frames producer (e.g. video, webcam, ...)
	auto wUserInput = std::make_shared<WUserInput>(FLAGS_image_dir);
    const auto workerInputOnNewThread = false;
    opWrapper.setWorkerInput(wUserInput, workerInputOnNewThread);
	*/
    // Add custom processing
    const auto workerProcessingOnNewThread = false;
    opWrapper.setWorkerPostProcessing(wUserPostProcessing, workerProcessingOnNewThread);
	/*
    // Add custom output
    const auto workerOutputOnNewThread = true;
    opWrapper.setWorkerOutput(wUserOutput, workerOutputOnNewThread);
	*/

	//oak 探测手但不画手
	FLAGS_hand = true;
	FLAGS_hand_render = 0;

	// Producer (use default to disable any input)
	const op::WrapperStructInput wrapperStructInput{ producerSharedPtr, FLAGS_frame_first, FLAGS_frame_last, FLAGS_process_real_time,
		FLAGS_frame_flip, FLAGS_frame_rotate, FLAGS_frames_repeat };


    // Configure OpenPose
    const op::WrapperStructPose wrapperStructPose{netInputSize, outputSize, keypointScale, FLAGS_num_gpu,
                                                  FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
                                                  op::flagsToRenderMode(FLAGS_render_pose), poseModel,
                                                  !FLAGS_disable_blending, (float)FLAGS_alpha_pose,
                                                  (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, FLAGS_model_folder,
                                                  heatMapTypes, heatMapScale, (float)FLAGS_render_threshold};
    // Face configuration (use op::WrapperStructFace{} to disable it)
    const op::WrapperStructFace wrapperStructFace{FLAGS_face, faceNetInputSize, op::flagsToRenderMode(FLAGS_face_render, FLAGS_render_pose),
                                                  (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
    // Hand configuration (use op::WrapperStructHand{} to disable it)
    const op::WrapperStructHand wrapperStructHand{FLAGS_hand, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
                                                  FLAGS_hand_tracking, op::flagsToRenderMode(FLAGS_hand_render, FLAGS_render_pose),
                                                  (float)FLAGS_hand_alpha_pose, (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};

    const op::WrapperStructOutput wrapperStructOutput{ !FLAGS_no_display, !FLAGS_no_gui_verbose, FLAGS_fullscreen, FLAGS_write_keypoint,
                                                      op::stringToDataFormat(FLAGS_write_keypoint_format), FLAGS_write_keypoint_json,
                                                      FLAGS_write_coco_json, FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video,
                                                      FLAGS_write_heatmaps, FLAGS_write_heatmaps_format};
    // Configure wrapper
    opWrapper.configure(wrapperStructPose, wrapperStructFace, wrapperStructHand, wrapperStructInput, wrapperStructOutput);
	//opWrapper.configure(wrapperStructPose, wrapperStructFace, wrapperStructHand, op::WrapperStructInput{}, wrapperStructOutput);
    // Set to single-thread running (e.g. for debugging purposes)
    // opWrapper.disableMultiThreading();

    op::log("Starting thread(s)", op::Priority::High);
    // Two different ways of running the program on multithread environment
    // // Option a) Recommended - Also using the main thread (this thread) for processing (it saves 1 thread)
    // // Start, run & stop threads
    opWrapper.exec();  // It blocks this thread until all threads have finished

    // Option b) Keeping this thread free in case you want to do something else meanwhile, e.g. profiling the GPU memory
    // // VERY IMPORTANT NOTE: if OpenCV is compiled with Qt support, this option will not work. Qt needs the main thread to
    // // plot visual results, so the final GUI (which uses OpenCV) would return an exception similar to:
    // // `QMetaMethod::invoke: Unable to invoke methods with return values in queued connections`
    // // Start threads
    // opWrapper.start();
    // // Profile used GPU memory
    //     // 1: wait ~10sec so the memory has been totally loaded on GPU
    //     // 2: profile the GPU memory
    // std::this_thread::sleep_for(std::chrono::milliseconds{1000});
    // op::log("Random task here...", op::Priority::High);
    // // Keep program alive while running threads
    // while (opWrapper.isRunning())
    //     std::this_thread::sleep_for(std::chrono::milliseconds{33});
    // // Stop and join threads
    // op::log("Stopping thread(s)", op::Priority::High);
    // opWrapper.stop();

    // Measuring total time
    const auto now = std::chrono::high_resolution_clock::now();
    const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerBegin).count() * 1e-9;
    const auto message = "Real-time pose estimation demo successfully finished. Total time: " + std::to_string(totalTimeSec) + " seconds.";
    op::log(message, op::Priority::High);

	stopConnect();

    return 0;
}

int main(int argc, char *argv[])
{
    // Initializing google logging (Caffe uses it for logging)
    google::InitGoogleLogging("openPoseTutorialWrapper2");

    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPoseTutorialWrapper2
    return openPoseTutorialWrapper2();
}
