import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

class BostonNode(Node):
    def __init__(self):
        super().__init__('boston_node')
        self.get_logger().info('Boston Housing Linear Regression Node started.')

        # Publishers
        self.mae_publisher = self.create_publisher(Float64, 'boston_mae', 10)
        self.results_publisher = self.create_publisher(String, 'boston_results', 10)

        # Load dataset
        try:
            boston = load_boston()
        except Exception as e:
            self.get_logger().error(f"Failed to load Boston dataset: {e}")
            return

        X = boston.data
        y = boston.target

        # Use just one feature (RM: average number of rooms)
        X_feature = X[:, 5].reshape(-1, 1)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        coef = model.coef_[0]
        intercept = model.intercept_

        self.get_logger().info(f'MAE: {mae:.2f}, R²: {r2:.2f}')
        self.get_logger().info(f'Coefficient (m): {coef:.2f}, Intercept (b): {intercept:.2f}')

        # Publish MAE as Float64
        mae_msg = Float64()
        mae_msg.data = mae
        self.mae_publisher.publish(mae_msg)

        # Publish summary as String
        result_msg = String()
        result_msg.data = f"MAE: {mae:.2f}, R²: {r2:.2f}, m: {coef:.2f}, b: {intercept:.2f}"
        self.results_publisher.publish(result_msg)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, color='blue', label='Training Data', alpha=0.5)
        plt.scatter(X_test, y_test, color='green', label='Test Data (Actual)', alpha=0.7)

        # Sort for line plotting
        X_all = np.vstack((X_train, X_test))
        X_line = np.linspace(X_all.min(), X_all.max(), 100).reshape(-1, 1)
        y_line = model.predict(X_line)

        plt.plot(X_line, y_line, color='red', label='Regression Line')
        plt.xlabel('Average Number of Rooms (RM)')
        plt.ylabel('House Price ($1000s)')
        plt.title('Boston Housing: Linear Regression')
        plt.legend()
        plt.grid(True)

        plot_path = '/home/anild/ros2_ws/src/linreg_analysis/boston_regression_plot_full.png'
        plt.savefig(plot_path)
        plt.close()
        self.get_logger().info(f'Plot saved to: {plot_path}')

        self.create_timer(1.0, lambda: None)

def main(args=None):
    rclpy.init(args=args)
    node = BostonNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

