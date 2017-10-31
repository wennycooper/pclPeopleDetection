#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h> 
#include <boost/foreach.hpp>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>

void point_picking();
void process_cloud();

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// PCL viewer //
pcl::visualization::PCLVisualizer viewer("PCL Viewer");

// Mutex: //
boost::mutex cloud_mutex;

enum { COLS = 640, ROWS = 480 };

PointCloud::Ptr cloud (new PointCloud);
bool new_cloud_available_flag = false;
bool first_cloud = true;

pcl::people::PersonClassifier<pcl::RGB> person_classifier;
pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object

std::string svm_filename = "/home/kkuei/catkin_ws/src/pclPeopleDetection/src/trainedLinearSVMForPeopleDetectionWithHOG.yaml";
float min_confidence = -1.5;
float min_height = 1.0;
float max_height = 2.3;
float voxel_size = 0.06;
Eigen::Matrix3f rgb_intrinsics_matrix;
Eigen::VectorXf ground_coeffs;

void callback(const PointCloud::ConstPtr& callback_cloud)
{
  cloud_mutex.lock();
  *cloud = *callback_cloud;
  new_cloud_available_flag = true;
  cloud_mutex.unlock(); 
  
  if (first_cloud) {
    first_cloud = false;
    //point_picking();
    //return;  
  } else {
//    std::cout << "more clouds..." << std::endl;
    process_cloud();
  }
}

struct callback_args{
  // structure used to pass arguments to the callback function
  PointCloud::Ptr clicked_points_3d;
  pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};

void
pp_callback (const pcl::visualization::PointPickingEvent& event, void* args)
{
  struct callback_args* data = (struct callback_args *)args;
  if (event.getPointIndex () == -1)
    return;
  PointT current_point;
  event.getPoint(current_point.x, current_point.y, current_point.z);
  data->clicked_points_3d->points.push_back(current_point);
  // Draw clicked points in red:
  pcl::visualization::PointCloudColorHandlerCustom<PointT> red (data->clicked_points_3d, 255, 0, 0);
  data->viewerPtr->removePointCloud("clicked_points");
  data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
  data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
  std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;
}


void point_picking()
{
  cloud_mutex.lock ();    // for not overwriting the point cloud

  // Display pointcloud:
  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
  viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
  viewer.setCameraPosition(0,0,-2,0,-1,0,0);

  // Add point picking callback to viewer:
  struct callback_args cb_args;
  PointCloud::Ptr clicked_points_3d (new PointCloud);
  cb_args.clicked_points_3d = clicked_points_3d;
  cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(&viewer);
  viewer.registerPointPickingCallback (pp_callback, (void*)&cb_args);
  std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;

  // Spin until 'Q' is pressed:
  viewer.spin();
  std::cout << "done." << std::endl;
  
  cloud_mutex.unlock ();  

  // Ground plane estimation:
  Eigen::VectorXf ground_coeffs;
  ground_coeffs.resize(4);
  std::vector<int> clicked_points_indices;
  for (unsigned int i = 0; i < clicked_points_3d->points.size(); i++)
    clicked_points_indices.push_back(i);
  pcl::SampleConsensusModelPlane<PointT> model_plane(clicked_points_3d);
  model_plane.computeModelCoefficients(clicked_points_indices,ground_coeffs);
  std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;


}

void
process_cloud()
{
  // Perform people detection on the new cloud:
  std::vector<pcl::people::PersonCluster<PointT> > clusters;   // vector containing persons clusters
  people_detector.setInputCloud(cloud);
  people_detector.setGround(ground_coeffs);                    // set floor coefficients
  people_detector.compute(clusters);                           // perform people detection

  ground_coeffs = people_detector.getGround();                 // get updated floor coefficients

  // Draw cloud and people bounding boxes in the viewer:
  //viewer.setCameraPosition(0,0,-2,0,-1,0,0);
  viewer.removeAllPointClouds();
  viewer.removeAllShapes();
  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
  viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
  
  unsigned int k = 0;
  for(std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
  {
    if(it->getPersonConfidence() > min_confidence)             // draw only people with confidence above a threshold
    {
      // draw theoretical person bounding box in the PCL viewer:
      it->drawTBoundingBox(viewer, k);
      k++;
    }
  }
  if (k>0) {
    std::cout << k << " people found" << std::endl;
  }

  viewer.spinOnce();
}



int main(int argc, char** argv)
{
  ros::init(argc, argv, "pclPeopleDetection");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe<PointCloud>("/camera/depth_registered/points", 1, callback);

  ground_coeffs.resize(4);
// coeffs in officei: -0.0323267 0.999054 -0.0291018 -0.947593
  ground_coeffs << -0.0323267, 0.999054, -0.0291018, -0.947593;

  rgb_intrinsics_matrix << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics



  // Create classifier for people detection:  
//  pcl::people::PersonClassifier<pcl::RGB> person_classifier;
  person_classifier.loadSVMFromFile(svm_filename);   // load trained SVM

  // People detection app initialization:
  // pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object
  people_detector.setVoxelSize(voxel_size);                        // set the voxel size
  people_detector.setIntrinsics(rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
  people_detector.setClassifier(person_classifier);                // set person classifier
  people_detector.setHeightLimits(min_height, max_height);         // set person classifier
//  people_detector.setSensorPortraitOrientation(true);             // set sensor orientation to vertical

  // set camera default position
  viewer.setCameraPosition(0,0,-2,0,-1,0,0);

  ros::spin();

  return 0;
}
