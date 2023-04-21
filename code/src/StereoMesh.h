#ifndef __STEREO_MESH_H
#define __STEREO_MESH_H
#include <VVRScene/canvas.h>
#include <VVRScene/mesh.h>
#include <VVRScene/settings.h>
#include <VVRScene/utils.h>
#include <MathGeoLib.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include <fstream>
#include <cstring>
#include <string>
#include <set>

#define DEFAULT				0.535353
#define FLAG_SHOW_AXES	       1
#define FLAG_SHOW_WIRE	       2
#define FLAG_SHOW_SOLID	       4
#define FLAG_SHOW_NORMALS      8
#define FLAG_SHOW_PLANE       16
#define FLAG_SHOW_AABB        32
#define FLAG_RANSAC           64
#define FLAG_DRAW_TRIANGLES  128
#define FLAG_SHOW_SIMILARITY 256
#define FLAG_SHOW_ORIGINAL   512
#define FLAG_GROUND_Z_LIM   1024
#define GROUND_RATIO        0.02
#define PROTOTYPES_N           4

#define BLUR_CANNY             0
#define KERNEL_SIZE            3
#define LOW_THRES              0
#define HIGH_THRES           100
#define RHO                  1.0
#define THETA                180
#define THRES                141
#define SRN                    0
#define STN                    0
#define THETA_MIN              0
#define THETA_MAX              1
#define MIN_LENGTH           150
#define MAX_GAP                5

using namespace cv;
namespace vvr {
	std::string base_dir = "../resources/images/carla/";
	//std::string base_dir = "../resources/images/kitti/";
	std::string img_src = "Town05_01_07_2020_23_49_52_128525.png";
	///std::string img_src = "Town03_05_07_2020_12_31_51_016755.png";
	//std::string img_src = "0000000000.png";

	Mat image, image_color, image_denoised, image_color_denoised;
	Mat mask1;
	Mat image_masked, image_blured, image_g_blured, image_m_blured, image_b_filtered;
	Mat detected_edges;
	Mat detected_lines, detected_lines_P;
	Mat final_image;
	//cv::vector<cv::Vec2f> lines, linesP;

	class Mesh3DScene : public vvr::Scene
	{
	public:
		Mesh3DScene();
		const char* getName() const { return "3D Scene"; }
		void keyEvent(unsigned char key, bool up, int modif) override;
		
	private:
		void draw() override;
		void reset() override;
		void resize() override;
		void Tasks();
		void load_point_cloud(std::vector<vec>& pcl, std::string pth, bool is_prototype);
		void find_aabb();
		void remove_ground_z_lim(double z_lim);
		void ransac_algorithm(int max_iter, double dist_threshold);
		void dbscan_algorithm(double eps, int min_pts);
		void find_neighbors(double d, int index, std::vector<int>& N);
		double compare_pcl(std::vector<int> pcl1, std::vector<vec> pcl2);
		//*/
		void triangulation();
		void find_triangle_neighbors(double d, int clstr, int index, std::vector<int>& N);
		int bnr_search2(std::vector<std::vector<double>>& arr, int l, int h, double val);
		void delete_double_triangles();
		bool is_same_triangle(Triangle3D& t1, Triangle3D& t2);
		bool is_same_point(vec p1, vec p2);
		//*/
		
	private:
		int m_style_flag;
		float m_plane_d;
		math::Plane m_plane;
		vvr::Canvas2D m_canvas;
		vvr::Colour m_obj_col;
		vvr::Box3D aabb;
		math::vec center_mass;
		std::vector<vec> point_cloud;
		std::vector<vec> point_cloud_original;
		std::vector<std::vector<vec>> prototypes;
		std::vector<int> label;
		std::vector<vvr::Colour> colors;
		std::vector<std::vector<int> > cluster;
		std::vector<int> ground_ransac;
		std::vector<int> ground_z_lim;
		std::vector<std::vector<double>> similarity;
		std::vector<vvr::Triangle3D> triangles;
	};
}
#endif // __VVR_LAB0_H