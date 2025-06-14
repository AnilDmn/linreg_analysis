import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

class HeightWeightNode(Node):
    def __init__(self):
        super().__init__('height_weight_node')
        self.get_logger().info("Starting linear regression on height vs weight...")

        # Publishers
        self.results_publisher = self.create_publisher(String, 'height_weight_results', 10)
        self.mae_publisher = self.create_publisher(Float64, 'height_weight_mae', 10)

        # Load dataset
        try:
            df = pd.read_csv('/home/anild/ros2_ws/src/linreg_analysis/data/new_height_weight.csv')
        except Exception as e:
            self.get_logger().error(f"Error loading CSV: {e}")
            return

        X = df[['Height']]
        y = df['Weight']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        m = model.coef_[0]
        b = model.intercept_

        # Logging
        self.get_logger().info(f"MAE: {mae:.2f}, R²: {r2:.2f}")
        self.get_logger().info(f"Model Coefficient (m): {m:.2f}")
        self.get_logger().info(f"Model Intercept (b): {b:.2f}")

        # Publish metrics as String
        results_msg = String()
        results_msg.data = f"MAE: {mae:.2f}, R²: {r2:.2f}, m: {m:.2f}, b: {b:.2f}"
        self.results_publisher.publish(results_msg)

        # Publish MAE as Float64
        mae_msg = Float64()
        mae_msg.data = mae
        self.mae_publisher.publish(mae_msg)
        self.get_logger().info(f'Published MAE: {mae:.2f} to topic /height_weight_mae')

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, label='Training Data', color='blue', alpha=0.5)
        plt.scatter(X_test, y_test, label='Test Data (Actual)', color='green', alpha=0.7)
        plt.scatter(X_test, predictions, label='Test Data (Predicted)', color='red', marker='x')

        x_line = sorted(X['Height'])
        x_line_array = np.array(x_line).reshape(-1, 1)
        y_line = model.predict(x_line_array)
        plt.plot(x_line, y_line, label='Regression Line', color='black', linestyle='--')

        plt.xlabel('Height (cm)')
        plt.ylabel('Weight (kg)')
        plt.title('Linear Regression: Height vs Weight')
        plt.legend()
        plt.grid(True)

        plt.savefig('/home/anild/ros2_ws/src/linreg_analysis/height_weight_plot_full.png')
        self.get_logger().info("Prediction plot saved successfully.")

        # Dummy timer to keep node alive for rqt_graph
        self.create_timer(1.0, lambda: None)

def main(args=None):
    rclpy.init(args=args)
    node = HeightWeightNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

