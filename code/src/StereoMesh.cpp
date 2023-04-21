#include "StereoMesh.h"

using namespace cv;
using namespace vvr;
using namespace std;
/*/
int size_canny_slider = 3;
int low_thres_canny_slider = 0;
int high_thres_canny_slider = 100;
int kernel_canny_slider = 3;
int blur_canny_box = 1;
double rho_hl_slider = 1.0;
int theta_hl_slider = 180;
int thres_hl_slider = 141;
double srn_hl_slider = 0;
double stn_hl_slider = 0;
int min_theta_hl_slider = 0;
int max_theta_hl_slider = 10;
int s1 = 150;
int s2 = 5;
//*/
vector<Vec2f> lines;
vector<Vec4i> linesP;

static void edge_detection2(int, void*);
void edge_detection();

int main(int argc, char* argv[]) {
	cv::namedWindow("Initial Image");
	cv::namedWindow("Image Denoised");
	cv::namedWindow("Line Detection:");
	cv::namedWindow("Image Masked");
	cv::namedWindow("Image Grayscale");


	image_color = imread(base_dir + img_src, cv::IMREAD_COLOR); // load image
	cvtColor(image_color, image, COLOR_BGR2GRAY); // convert image to grayscale

	//If not loaded, return
	if (image_color.empty()) {
		cout << "Image not Loaded: Incorrect filepath" << endl;
		system("pause");
		return -1;
	}
	// Denoise image
	fastNlMeansDenoising(image, image_denoised);
	fastNlMeansDenoisingColored(image_color, image_color_denoised);
	fastNlMeansDenoisingColored(image_color, final_image);
	//fastNlMeansDenoising(image_color, image_color_denoised);

	cv::imshow("Initial Image", image_color);
	cv::imshow("Image Grayscale", image);
	cv::imshow("Image Denoised", image_color_denoised);

	edge_detection();
	/*/
	createTrackbar("Slider1:", "Sliders", &size_canny_slider, 3, edge_detection2);
	createTrackbar("Slider2:", "Sliders", &low_thres_canny_slider, 100, edge_detection2);
	createTrackbar("Slider3:", "Sliders", &high_thres_canny_slider, 300, edge_detection2);
	//createTrackbar("Slider4:", "Sliders", &kernel_canny_slider, 7, edge_detection2);
	createTrackbar("Slider5:", "Sliders", &blur_canny_box, 1, edge_detection2);
	//createTrackbar("Slider6:", "Sliders", &rho_hl_slider, 3, edge_detection2);
	createTrackbar("Slider7:", "Sliders", &theta_hl_slider, 360, edge_detection2);
	createTrackbar("Slider8:", "Sliders", &thres_hl_slider, 300, edge_detection2);
	//createTrackbar("Slider9:", "Sliders", &srn_hl_slider, 1, edge_detection2);
	//createTrackbar("Slider10:", "Sliders", &stn_hl_slider, 1, edge_detection2);
	createTrackbar("Slider11:", "Sliders", &min_theta_hl_slider, 10, edge_detection2);
	createTrackbar("Slider12:", "Sliders", &max_theta_hl_slider, 10, edge_detection2);
	createTrackbar("Slider13:", "Sliders", &s1, 200, edge_detection2);
	createTrackbar("Slider14:", "Sliders", &s2, 100, edge_detection2);
	
	edge_detection2(0, 0);
	//*/
	cout<< "b --> for lidar space" << endl << endl;

	while (true) {
		if (waitKey(10) == 'b')
			break;
	}

	//Clean up
	//destroyAllWindows();

	// Start Scene Class
	try {
		return vvr::mainLoop(argc, argv, new Mesh3DScene);
	}
	catch (std::string exc) {
		cerr << exc << endl;
		return 1;
	}
	catch (...) {
		cerr << "Unknown exception" << endl;
		return 1;
	}
}

void edge_detection() {
	mask1 = Mat::zeros(image.size(), image.type());
	Point pts[6] = {
		Point(0,image.size().height),
		Point(0,(2.0 / 3.0) * image.size().height),
		Point((1.0 / 3.0) * image.size().width,0.5 * image.size().height),
		Point((2.0 / 3.0) * image.size().width,0.5 * image.size().height),
		Point(image.size().width,(2.0 / 3.0) * image.size().height),
		Point(image.size().width,image.size().height)
	};

	fillConvexPoly(mask1, pts, 6, Scalar(1)); // create mask
	image_denoised.copyTo(image_masked, mask1); // mask image
	imshow("Image Masked", image_masked);
	// add optional filter
	blur(image_masked, image_blured, Size(KERNEL_SIZE, KERNEL_SIZE));
	GaussianBlur(image_masked, image_g_blured, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0);
	medianBlur(image_masked, image_m_blured, KERNEL_SIZE);
	bilateralFilter(image_masked, image_b_filtered, KERNEL_SIZE, KERNEL_SIZE * 2, KERNEL_SIZE / 2);

	if (BLUR_CANNY) {
		Canny(image_blured, detected_edges, LOW_THRES, HIGH_THRES, KERNEL_SIZE);
	}
	else {
		Canny(image_masked, detected_edges, LOW_THRES, HIGH_THRES, KERNEL_SIZE);
	}
	///cvtColor(detected_edges, detected_lines, COLOR_GRAY2BGR);
	/// HL
	//HoughLines(detected_edges, lines, RHO, CV_PI / THETA, THRES, SRN, STN, THETA_MIN * CV_PI, THETA_MAX * CV_PI);

	/// HLP
	///cvtColor(image_masked, detected_lines_P, COLOR_GRAY2BGR);
	HoughLinesP(detected_edges, linesP, RHO, CV_PI / THETA, THRES, MIN_LENGTH, MAX_GAP);

	/// cluster lines missing function (keep lines that have two or more lines in its cluster)

	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		line(final_image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
	cv::imshow("Line Detection:", final_image);
}

//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------

Mesh3DScene::Mesh3DScene() {
	//! Load settings.
	vvr::Shape::DEF_LINE_WIDTH = 4;
	vvr::Shape::DEF_POINT_SIZE = 5;
	m_perspective_proj = true;
	m_bg_col = Colour("768E77");
	m_obj_col = Colour("454545");

	reset();
}

void Mesh3DScene::load_point_cloud(std::vector<vec>& pcl, std::string pth, bool is_prototype) {
	//! loads main scene point cloud and prototypes for comparison

	center_mass = vec(0, 0, 0);

	fstream input(pth.c_str(), ios::in | ios::binary);
	if (!input.good()) {
		cerr << "Could not read file: " << pth << endl;
		exit(EXIT_FAILURE);
	}
	input.seekg(0, ios::beg);
	int i;
	for (i = 0; input.good() && !input.eof(); i++) {
		vec point;
		float intensity;
		input.read((char*)&point.x, 3 * sizeof(float));
		input.read((char*)&intensity, sizeof(float));
		center_mass += point;
		pcl.push_back(point);
		label.push_back(-1); //default point label (=unused)
	}
	input.close();
	
	center_mass /= pcl.size();
	// move to point of interest
	for (int i=0;i< pcl.size();i++) {
		pcl[i].x= pcl[i].x- center_mass[0];
		pcl[i].y= pcl[i].y- center_mass[1];
		pcl[i].z= pcl[i].z- center_mass[2];
	}
}

void Mesh3DScene::find_aabb() {
	//! find bounding-box of point cloud

	const unsigned N = point_cloud_original.size();

	double max_x = point_cloud_original[0].x;
	double max_y = point_cloud_original[0].y;
	double max_z = point_cloud_original[0].z;
	double min_x = point_cloud_original[0].x;
	double min_y = point_cloud_original[0].y;
	double min_z = point_cloud_original[0].z;

	for (int i = 0; i < N; i++) {
		if (point_cloud_original[i].x > max_x)
			max_x = point_cloud_original[i].x;
		if (point_cloud_original[i].y > max_y)
			max_y = point_cloud_original[i].y;
		if (point_cloud_original[i].z > max_z)
			max_z = point_cloud_original[i].z;
		if (point_cloud_original[i].x < min_x)
			min_x = point_cloud_original[i].x;
		if (point_cloud_original[i].y < min_y)
			min_y = point_cloud_original[i].y;
		if (point_cloud_original[i].z < min_z)
			min_z = point_cloud_original[i].z;
	}
	aabb.x1 = min_x;
	aabb.y1 = min_y;
	aabb.z1 = min_z;
	aabb.x2 = max_x;
	aabb.y2 = max_y;
	aabb.z2 = max_z;

	//! Define plane
	m_plane_d = aabb.z1;
	m_plane = Plane(vec(0, 0, 1).Normalized(), m_plane_d);
}

void Mesh3DScene::remove_ground_z_lim(double z_lim) {
	for (int i = 0; i < point_cloud_original.size(); i++) {
		if (point_cloud_original[i].z >= z_lim) {
			point_cloud.push_back(point_cloud_original[i]);
		}
		else {
			ground_z_lim.push_back(i);
		}
	}
}

void Mesh3DScene::ransac_algorithm(int max_iter, double dist_threshold) {
	//! ransac algorithm implementation to find plane -> ground
	//! 	
	while (max_iter) {
		max_iter -= 1;
		vector<int> inliers_index;

		int r = rand() % point_cloud_original.size();
		inliers_index.push_back(r);

		for (int k = 0; k < 2; k++) {
			int r = rand() % point_cloud_original.size();
			inliers_index.push_back(r);
		}
		double x1, y1, z1, x2, y2, z2, x3, y3, z3;
		double a, b, c, d;
		x1 = point_cloud_original[inliers_index[0]].x;
		y1 = point_cloud_original[inliers_index[0]].y;
		z1 = point_cloud_original[inliers_index[0]].z;
		x2 = point_cloud_original[inliers_index[1]].x;
		y2 = point_cloud_original[inliers_index[1]].y;
		z2 = point_cloud_original[inliers_index[1]].z;
		x3 = point_cloud_original[inliers_index[2]].x;
		y3 = point_cloud_original[inliers_index[2]].y;
		z3 = point_cloud_original[inliers_index[2]].z;

		a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
		b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
		c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
		d = -(a * x1 + b * y1 + c * z1);
		double plane_length = max(0.1,sqrt(a*a+b*b+c*c));

		for (int i = 0; i < point_cloud_original.size(); i++) {
			//*/
			bool found = false;
			for (int j = 0; j < inliers_index.size(); j++) {
				if (inliers_index[j] == i) {
					found = true;
					break;
				}
			}
			//*/
			if (found) {
				continue;
			}

			double x, y, z;
			x = point_cloud_original[i].x;
			y = point_cloud_original[i].y;
			z = point_cloud_original[i].z;

			double distance = abs(a*x + b*y + c*z + d)/plane_length;
			if (distance <= dist_threshold) {
				inliers_index.push_back(i);
			}
		}
		if (inliers_index.size() > ground_ransac.size()) {
			ground_ransac.clear();
			ground_ransac = inliers_index;
		}
	}
}

void Mesh3DScene::dbscan_algorithm(double eps, int min_pts) {
	//! dbscan algorithm implementation to find clusters
	//! 
	int c = 0;
	for (int i = 0; i < point_cloud.size(); i++) {
		///if (point_cloud[i].z < aabb.z1 + (aabb.z2 - aabb.z1) * GROUND_RATIO)
			///continue;
		if (label[i] != -1)
			continue;
		vector<int> neighbors;
		find_neighbors(eps, i, neighbors);
		if (neighbors.size() < min_pts) {
			label[i] = 0;
			continue;
		}

		c += 1;
		label[i] = c;
		for (int j = 0; j < neighbors.size(); j++) {
			if (label[neighbors[j]] == 0)
				label[neighbors[j]] = c;
			if (label[neighbors[j]] != -1)
				continue;
			label[neighbors[j]] = c;
			vector<int> neighbors2;
			find_neighbors(eps, neighbors[j], neighbors2);
			if (neighbors.size() >= min_pts) {
				neighbors.insert(neighbors.begin(), neighbors2.begin(), neighbors2.end());
			}
		}
	}
	int number_of_clusters = *max_element(label.begin(), label.end());
	
	vector<vector<int> > clstr(number_of_clusters+1);
	
	for (int i = 0; i < point_cloud.size(); i++) {
		if (label[i] != -1) {
			clstr[label[i]].push_back(i);
		}
	}
	cluster = clstr;
}

void Mesh3DScene::find_neighbors(double d, int index, std::vector<int>& N) {
	//! finds neighbors of point in index <index> that have less distance than d
	//! 
	for (int i = 0; i < point_cloud.size(); i++) {
		///if (point_cloud[i].z < aabb.z1 + (aabb.z2 - aabb.z1) * GROUND_RATIO)
			///continue;
		if (i == index) {
			continue;
		}
		
		double a, b, c;
		a = point_cloud[i].x - point_cloud[index].x;
		b = point_cloud[i].y - point_cloud[index].y;
		c = point_cloud[i].z - point_cloud[index].z;

		if (a>d || b>d ||c>d)
			continue;

		double dist = sqrt(a*a + b*b +c*c);
		if (dist <= d) {
			N.push_back(i);
		}
	}
}

double Mesh3DScene::compare_pcl(std::vector<int> pcl1, std::vector<vec> pcl2) {
	//! find similarity (0->1) between two point clouds with custom metrics
	//!  
	double smlrt = 0.0;
	double sm1, sm2, sm3, sm3a, sm3b;
	double z1_min, z2_min, z1_max, z2_max;
	double r1_mean, r2_mean, r1_s, r2_s, r1_max, r2_max;
	vector<double> r1,r2;

	sm1 = 0.0;
	sm2 = 0.0;
	sm3 = 0.0;
	sm3a = 0.0;
	sm3b = 0.0;
	z1_min = point_cloud[pcl1[0]].z;
	z1_max = point_cloud[pcl1[0]].z;
	z2_min = pcl2[0].z;
	z2_max = pcl2[0].z;
	r1_max = 0;
	r2_max = 0;
	r1_mean = 0;
	r2_mean = 0;
	r1_s = 0;
	r2_s = 0;
	
	for (int i = 0; i < pcl1.size(); i++) {
		double z = point_cloud[pcl1[i]].z;
		double x = point_cloud[pcl1[i]].x;
		double y = point_cloud[pcl1[i]].y;
		if (z < z1_min)
			z1_min = z;
		if (z > z1_max)
			z1_max = z;
		double r = sqrt(x * x + y * y);
		r1.push_back(r);
		r1_mean += r;
		if (r > r1_max)
			r1_max = r;
	}
	r1_mean/= r1.size();
	for (int i = 0; i < r1.size(); i++) {
		r1_s += (r1[i]-r1_mean) * (r1[i] - r1_mean);
	}
	r1_s /= r1.size();

	for (int j = 0; j < pcl2.size(); j++) {
		if (pcl2[j].z < z2_min)
			z2_min = pcl2[j].z;
		if (pcl2[j].z > z2_max)
			z2_max = pcl2[j].z;
		double rr =sqrt(pcl2[j].x * pcl2[j].x + pcl2[j].y * pcl2[j].y);
		r2.push_back(rr);
		r2_mean += rr;
		if (rr > r2_max)
			r2_max = rr;
	}
	r2_mean /= r2.size();
	for (int j = 0; j < r2.size(); j++) {
		r2_s += (r2[j] - r2_mean) * (r2[j] - r2_mean);
	}
	r2_s /= r2.size();

	sm1 = 1.0 - abs(z1_min - z2_min)/max(abs(z1_min-z2_max),abs(z2_min-z1_max));
	sm2 = 1.0 - abs((z1_max - z1_min) - (z2_max - z2_min))/max((z1_max - z1_min),(z2_max - z2_min));
	///sm2 = 1.0 - abs((z1_max - z1_min) - (z2_max - z2_min))/((z1_max - z1_min) + (z2_max - z2_min));

	sm3a = 1.0 - abs(r1_s - r2_s)/max(r1_s,r2_s);
	///sm3b = 1.0 - abs(r1_s - r2_s)/(r1_s + r2_s);
	sm3b = 1.0 - abs(r1_max-r2_max)/max(r1_max,r2_max);
	///sm3c = abs(r1_max-r2_max)/(r1_max+r2_max);

	sm3 = (sm3a + sm3b)/2.0;

	smlrt = (3.0*sm1 + 3.0*sm2 + 4.0*sm3) / 10.0;
	//smlrt = (sm1 + sm2 + sm3) / 3.0;
	return smlrt;
}
//*/
void Mesh3DScene::triangulation() {
	//! triangulate clusters 
	//! make 4 triangles for every 4 nearest neighbors of each point
	//! 
	vec p, p1, p2, p3, p4;
	for (int i = 0; i < cluster.size(); i++) {
		if (i != 155)
			continue;
		for (int j = 0; j < cluster[i].size(); j++) {
			vector<int> neighbors;
			find_triangle_neighbors(15.0, i, cluster[i][j], neighbors);
			//*/
			p = point_cloud[cluster[i][j]];
			p1 = point_cloud[neighbors[0]];
			p2 = point_cloud[neighbors[1]];
			p3 = point_cloud[neighbors[2]];
			p4 = point_cloud[neighbors[3]];
			triangles.push_back(Triangle3D(p.x, p.y, p.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, Colour::green));
			triangles.push_back(Triangle3D(p.x, p.y, p.z, p1.x, p1.y, p1.z, p3.x, p3.y, p3.z, Colour::green));
			triangles.push_back(Triangle3D(p.x, p.y, p.z, p4.x, p4.y, p4.z, p2.x, p2.y, p2.z, Colour::green));
			triangles.push_back(Triangle3D(p.x, p.y, p.z, p3.x, p3.y, p3.z, p4.x, p4.y, p4.z, Colour::green));
			//*/

			/*/
			if (neighbors.size() < 2)
				continue;
			p = point_cloud[cluster[i][j]];
			p1 = point_cloud[neighbors[0]];
			p2 = point_cloud[neighbors[1]];
			triangles.push_back(Triangle3D(p.x, p.y, p.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, Colour::green));
			if (neighbors.size() == 3) {
				p3 = point_cloud[neighbors[2]];
				triangles.push_back(Triangle3D(p.x, p.y, p.z, p1.x, p1.y, p1.z, p3.x, p3.y, p3.z, Colour::green));
			}
			if (neighbors.size() == 4) {
				p3 = point_cloud[neighbors[2]];
				p4 = point_cloud[neighbors[3]];
				triangles.push_back(Triangle3D(p.x, p.y, p.z, p4.x, p4.y, p4.z, p2.x, p2.y, p2.z, Colour::green));
				triangles.push_back(Triangle3D(p.x, p.y, p.z, p3.x, p3.y, p3.z, p4.x, p4.y, p4.z, Colour::green));
			}
			//*/
		}
	}
	delete_double_triangles();
}

void Mesh3DScene::find_triangle_neighbors(double d, int clstr, int index, std::vector<int>& N) {
	//! find the 4 nearest neighbors (in cluster) of point index
	//! 
	vector<vector<double>> N_;
	bool first_time = true;
	for (int i = 0; i < cluster[clstr].size(); i++) {
		vector<double> pn;
		int k = index;
		int l = cluster[clstr][i];
		if (k == l) {
			continue;
		}
		double a, b, c;
		a = point_cloud[k].x - point_cloud[l].x;
		b = point_cloud[k].y - point_cloud[l].y;
		c = point_cloud[k].z - point_cloud[l].z;
		//if (a > d || b > d || c > d)
			//continue;

		double dist = sqrt(a * a + b * b + c * c);
		//if (dist > d)
			//continue;

		pn.push_back(l);
		pn.push_back(dist);
		if (first_time) {
			first_time = false;
			N_.push_back(pn);
		}
		int loc = bnr_search2(N_,0,N_.size()-1,dist);
		N_.insert(N_.begin() + loc, pn);
	}
	for (int j = 0; j < N_.size(); j++) {
		if (j == 4)
			break;
		N.push_back(int(N_[j][0]));
	}
}

int Mesh3DScene::bnr_search2(std::vector<std::vector<double>>& arr, int l, int h, double val) {
	//! binary search implementation for a 2d array 
	//! 
	if (h <= l) {
		return (arr[l][1] < val) ? (l + 1) : l;
	}
	int mid = (l + h) / 2;
	if (arr[mid][1] == val)
		return mid + 1;
	if (arr[mid][1] > val)
		return bnr_search2(arr, l, mid - 1, val);

	return bnr_search2(arr, mid + 1, h, val);
}

void Mesh3DScene::delete_double_triangles() {
	//! deletes double triangles made in triangulation
	//! 
	for (int i = 0; i < triangles.size(); i++) {
		for (int j = i; j < triangles.size(); j++) {
			if (is_same_triangle(triangles[i],triangles[j])) {
				triangles.erase(triangles.begin() + j);
			}
		}
	}
}

bool Mesh3DScene::is_same_triangle(Triangle3D& t1, Triangle3D& t2) {
	//! searches if two triangles have same points (=same triangles)
	//! 
	vec t1_p1 = vec(t1.x1, t1.y1, t1.z1);
	vec t1_p2 = vec(t1.x2, t1.y2, t1.z2);
	vec t1_p3 = vec(t1.x3, t1.y3, t1.z3);
	vec t2_p1 = vec(t2.x1, t2.y1, t2.z1);
	vec t2_p2 = vec(t2.x2, t2.y2, t2.z2);
	vec t2_p3 = vec(t2.x3, t2.y3, t2.z3);

	if (is_same_point(t1_p1, t2_p1) && is_same_point(t1_p2, t2_p2) && is_same_point(t1_p3, t2_p3))
		return true;
	else if (is_same_point(t1_p1, t2_p1) && is_same_point(t1_p2, t2_p3) && is_same_point(t1_p3, t2_p2))
		return true;
	else if (is_same_point(t1_p1, t2_p2) && is_same_point(t1_p2, t2_p3) && is_same_point(t1_p3, t2_p1))
		return true;
	else if (is_same_point(t1_p1, t2_p2) && is_same_point(t1_p2, t2_p1) && is_same_point(t1_p3, t2_p3))
		return true;
	else if (is_same_point(t1_p1, t2_p3) && is_same_point(t1_p2, t2_p1) && is_same_point(t1_p3, t2_p2))
		return true;
	else if (is_same_point(t1_p1, t2_p3) && is_same_point(t1_p2, t2_p2) && is_same_point(t1_p3, t2_p1))
		return true;
	else
		return false;
}

bool Mesh3DScene::is_same_point(vec p1, vec p2) {
	//! search if two points are same (=same coordinates)
	//! 
	if (p1.x == p2.x && p1.y == p2.y && p1.z == p2.z)
		return true;
	else
		return false;
}

void Mesh3DScene::reset() {
	Scene::reset();

	//! Define what will be vissible by default
	m_style_flag = 0;
	m_style_flag |= FLAG_SHOW_SOLID;
	m_style_flag |= FLAG_SHOW_WIRE;
	m_style_flag |= FLAG_SHOW_AXES;
	//m_style_flag |= FLAG_RANSAC;
	//m_style_flag |= FLAG_SHOW_AABB;
	//m_style_flag |= FLAG_SHOW_PLANE;
	//m_style_flag |= FLAG_DRAW_TRIANGLES;
	//m_style_flag |= FLAG_SHOW_SIMILARITY;
	//m_style_flag |= FLAG_SHOW_ORIGINAL;
	//m_style_flag |= FLAG_GROUND_Z_LIM;
}

void Mesh3DScene::resize() {
	static bool first_pass = true;

	if (first_pass)	{
		Tasks();

		std::cout << "b->show AABB"
			<< std::endl << "o --> show original point cloud"
			<< std::endl << "z --> show ground estimated with z limit"
			<< std::endl << "c --> show ground estimated with RANSAC algorithm"
			<< std::endl << "s --> show similarity of objects"
			<< std::endl << "t --> show triangles of triangulation"
			<< std::endl << endl;

		first_pass = false;
	}
}

void Mesh3DScene::Tasks() {
	center_mass = vec(0, 0, 0);
	// make a color vector
	colors.push_back(vvr::Colour::black);
	colors.push_back(vvr::Colour::darkOrange);
	colors.push_back(vvr::Colour::blue);
	colors.push_back(vvr::Colour::green);
	colors.push_back(vvr::Colour::red);
	colors.push_back(vvr::Colour::cyan);
	colors.push_back(vvr::Colour::yellowGreen);
	colors.push_back(vvr::Colour::darkGreen);
	colors.push_back(vvr::Colour::darkRed);
	colors.push_back(vvr::Colour::yellow);
	colors.push_back(vvr::Colour::white);
	colors.push_back(vvr::Colour::orange);
	colors.push_back(vvr::Colour::magenta);
	//colors.push_back(vvr::Colour::grey);

	string lidar_path = getBasePath() + "resources/lidar/";
	string prototype_path = getBasePath() + "resources/prototypes/";
	//load main scene point cloud
	string pth = lidar_path + "pointcloud_00529_.bin";
	load_point_cloud(point_cloud_original, pth, false);
	//load prototypes for comparison
	std::vector<vec> prototype_;
	pth = prototype_path + "car_00064_2257.bin";
	load_point_cloud(prototype_, pth, true);
	prototypes.push_back(prototype_);
	pth = prototype_path + "car_01143_2317.bin";
	load_point_cloud(prototype_, pth, true);
	prototypes.push_back(prototype_);
	pth = prototype_path + "pedestrian_00231_1938.bin";
	load_point_cloud(prototype_, pth, true);
	prototypes.push_back(prototype_);
	pth = prototype_path + "pedestrian_01126_2023.bin";
	load_point_cloud(prototype_, pth, true);
	prototypes.push_back(prototype_);

	find_aabb();
	// remove ground with z threshold
	remove_ground_z_lim(aabb.z1 + (aabb.z2 - aabb.z1) * GROUND_RATIO);

	ransac_algorithm(500, (aabb.z2 - aabb.z1)* GROUND_RATIO/4.0);

	dbscan_algorithm((aabb.z2 - aabb.z1)*0.075, 10); //dbscan_algorithm((aabb.z2 - aabb.z1)*0.065, 8);
	// compute similarity of every cluster for each prototype
	for (int i = 0; i < cluster.size(); i++) {
		vector<double> s;
		for (int j = 0; j < prototypes.size(); j++) {
			double s_ = compare_pcl(cluster[i], prototypes[j]);
			s.push_back(s_);
		}
		similarity.push_back(s);
	}
	// triangulate clusters
	triangulation();
}

void Mesh3DScene::keyEvent(unsigned char key, bool up, int modif) {
	Scene::keyEvent(key, up, modif);
	key = tolower(key);

	switch (key) {
	case 'b': m_style_flag ^= FLAG_SHOW_AABB; break;
	case 'c': m_style_flag ^= FLAG_RANSAC; break;
	case 't': m_style_flag ^= FLAG_DRAW_TRIANGLES; break;
	case 's': m_style_flag ^= FLAG_SHOW_SIMILARITY; break;
	case 'o': m_style_flag ^= FLAG_SHOW_ORIGINAL; break;
	case 'z': m_style_flag ^= FLAG_GROUND_Z_LIM; break;
	}
}

void Mesh3DScene::draw() {
	//*/
	//! Draw point cloud
	//! 
	
	if (m_style_flag & FLAG_SHOW_ORIGINAL) {
		for (int i = 0; i < point_cloud_original.size(); i++) {
			Point3D(point_cloud_original[i].x, point_cloud_original[i].y, point_cloud_original[i].z, vvr::Colour::darkOrange).draw();
		}
	}
	else {
		if (m_style_flag & FLAG_RANSAC) {
			for (int i = 0; i < ground_ransac.size(); i++) {
				int j = ground_ransac[i];
				Point3D(point_cloud_original[j].x, point_cloud_original[j].y, point_cloud_original[j].z, vvr::Colour::darkOrange).draw();
			}
		}
		else {
			if (m_style_flag & FLAG_GROUND_Z_LIM) {
				for (int i = 0; i < ground_z_lim.size(); i++) {
					int j = ground_z_lim[i];
					Point3D(point_cloud_original[j].x, point_cloud_original[j].y, point_cloud_original[j].z, vvr::Colour::darkOrange).draw();
				}
			}
			else {
				for (int i = 0; i < point_cloud.size(); i++) {
					if (label[i] == 0) // remove noise
						continue;
					if (m_style_flag & FLAG_SHOW_SIMILARITY)
						Point3D(point_cloud[i].x, point_cloud[i].y, point_cloud[i].z, Colour(to_string(floor(similarity[label[i]][0] * 99)) + "0000")).draw();
					else
						Point3D(point_cloud[i].x, point_cloud[i].y, point_cloud[i].z, colors[label[i] % 10]).draw();
				}
			}
		}
	}
	//*/

	//! Draw AABB
	if (m_style_flag & FLAG_SHOW_AABB) {
		aabb.setColour(Colour::black);
		aabb.setTransparency(1);
		aabb.draw();
	}

	//! Draw plane
	if (m_style_flag & FLAG_SHOW_PLANE) {
		vvr::Colour colPlane(0x41, 0x14, 0xB3);
		float u = 40, v = 40;
		math::vec p0(m_plane.Point(-u, -v, math::vec(0, 0, 0)));
		math::vec p1(m_plane.Point(-u, v, math::vec(0, 0, 0)));
		math::vec p2(m_plane.Point(u, -v, math::vec(0, 0, 0)));
		math::vec p3(m_plane.Point(u, v, math::vec(0, 0, 0)));
		math2vvr(math::Triangle(p0, p1, p2), colPlane).draw();
		math2vvr(math::Triangle(p2, p1, p3), colPlane).draw();
	}

	//! Draw triangles
	//*/
	if (m_style_flag & FLAG_DRAW_TRIANGLES) {
		for (int i = 0; i < triangles.size(); i++) {
			triangles[i].draw();
		}
	}
	//*/
}