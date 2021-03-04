// Author: Jack Bonnell
// Company: Sundance Multiprocessor Technology Ltd
// Email: jack.b@sundance.com
//Website: www.sundance.com
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <iostream>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <vitis/ai/demo.hpp>
#include <vitis/ai/facedetect.hpp>

using namespace std;
int main(int argc, char * argv[]) try
{
    std::cout << "Realsense Vitis AI Demo" << std::endl;
    std::cout << "By Jack Bonnell" << std::endl; 
    std::cout << "Email: Jack.B@sundance.com" << std::endl;   
    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 424, 240,RS2_FORMAT_Z16,6);
    cfg.enable_stream(RS2_STREAM_COLOR, 320, 240,RS2_FORMAT_BGR8,6);
    // Start streaming with default recommended configuration
    rs2::pipeline_profile profile = pipe.start(cfg);
    // Each depth camera might have different units for depth pixels, so we get it here
    // Using the pipeline's profile, we can retrieve the device that the pipeline uses
    rs2::align align_to_color(RS2_STREAM_COLOR);
    //Pipeline could choose a device that does not have a color stream
    //If there is no color stream, choose to align depth to another stream
    rs2::device dev = profile.get_device();
    rs2::depth_sensor ds = dev.query_sensors().front().as<rs2::depth_sensor>();
    float scale = ds.get_depth_scale();
    float depth_clipping_distance = 1.f;
    using namespace cv;
    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);
   auto network = vitis::ai::FaceDetect::create(
                  "densebox_640_360",
                  true);
    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
	data = align_to_color.process(data);
	rs2::video_frame rgb = data.get_color_frame();
        uint8_t* rgb_data = (uint8_t*)rgb.get_data();
        rs2::depth_frame depth = data.get_depth_frame();
        uint16_t* depth_data = (uint16_t*)depth.get_data();
	
        // Query frame size (width and height)
        const int w = rgb.as<rs2::video_frame>().get_width();
        const int h = rgb.as<rs2::video_frame>().get_height();
        const int rgb_bpp = rgb.get_bytes_per_pixel();


	for (int y = 0; y < h; y++)
	    {
		auto depth_pixel_index = y * w;
		for (int x = 0; x < w; x++, ++depth_pixel_index)
		{
		    // Get the depth value of the current pixel
		    auto pixels_distance = scale * depth_data[depth_pixel_index];

		    // Check if the depth value is invalid (<=0) or greater than the threashold
		    if (pixels_distance <= 0.f || pixels_distance > depth_clipping_distance)
		    {
		        // Calculate the offset in other frame's buffer to current pixel
		        auto offset = depth_pixel_index * rgb_bpp;

		        // Set pixel to "background" color (0x999999)
		        std::memset(&rgb_data[offset], 0x99, rgb_bpp);
		    }
		}
	    }
        Mat image(Size(w, h), CV_8UC3, (void*)rgb_data, Mat::AUTO_STEP);
        //Resizing for densebox 640 by 360
        resize(image, image, Size(640,360));
        //The VITIS-AI MAGIC
	  auto face_results = network->run(image);
        //Drawing boxes around results
	for (const auto &r : face_results.rects) {
	  int x1 = r.x * image.cols;
	  int y1 = r.y * image.rows;
	  int x2 = x1 + (r.width * image.cols);
	  int y2 = y1 + (r.height * image.rows);
          Point pt1(x1, y1);
          // and its bottom right corner.
          Point pt2(x2, y2);
          cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0));

	}


        // Update the window with new data
        imshow(window_name, image);
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
