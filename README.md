# linreg_analysis

This is a ROS 2 Python package that performs linear regression analysis on different datasets using `scikit-learn`, and visualizes the results with `matplotlib`.

## Features

- **Boston Housing Dataset**  

- **Height vs Weight Dataset**  

- **Head Size vs Brain Weight Dataset**  

Each analysis:
- Trains a linear regression model
- Publishes Mean Absolute Error (MAE) and model summary to ROS 2 topics
- Saves the result as a plot image

## ROS 2 Topics

Each node publishes:
- `/[dataset]_mae` (`std_msgs/Float64`)
- `/[dataset]_results` (`std_msgs/String`)

## Dependencies

Make sure the following are installed:

```bash
sudo apt install ros-humble-rclpy python3-numpy python3-matplotlib
pip install pandas scikit-learn
