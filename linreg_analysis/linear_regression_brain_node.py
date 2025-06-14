import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

class BrainWeightNode(Node):
    def __init__(self):
        super().__init__('brain_weight_node')
        self.get_logger().info("Starting linear regression on head size vs brain weight...")

        # Publishers
        self.results_publisher = self.create_publisher(String, 'brain_weight_results', 10)
        self.mae_publisher = self.create_publisher(Float64, 'brain_weight_mae', 10)

        # Load dataset
        try:
            df = pd.read_csv('/home/anild/ros2_ws/src/linreg_analysis/data/HumanBrain_WeightandHead_size.csv')
        except Exception as e:
            self.get_logger().error(f"Error loading CSV: {e}")
            return

        X = df[['Head Size(cm^3)']]
        y = df['Brain Weight(grams)']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        coef = model.coef_[0]
        intercept = model.intercept_

        # Logging
        self.get_logger().info(f"MAE: {mae:.2f}, R²: {r2:.2f}")
        self.get_logger().info(f"Model Coefficient (m): {coef:.2f}")
        self.get_logger().info(f"Model Intercept (b): {intercept:.2f}")

        # Publish string summary
        result_msg = String()
        result_msg.data = f"MAE: {mae:.2f}, R²: {r2:.2f}, Coef: {coef:.2f}, Intercept: {intercept:.2f}"
        self.results_publisher.publish(result_msg)

        # Publish MAE as Float64
        mae_msg = Float64()
        mae_msg.data = mae
        self.mae_publisher.publish(mae_msg)
        self.get_logger().info(f'Published MAE: {mae:.2f} to topic /brain_weight_mae')

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, label='Training Data', color='blue', alpha=0.5)
        plt.scatter(X_test, y_test, label='Test Data (Actual)', color='green', alpha=0.7)
        plt.scatter(X_test, predictions, label='Test Data (Predicted)', color='red', marker='x')

        x_line = np.linspace(X.min()[0], X.max()[0], 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        plt.plot(x_line, y_line, color='black', linestyle='--', label='Regression Line')

        plt.xlabel('Head Size (cm³)')
        plt.ylabel('Brain Weight (g)')
        plt.title('Linear Regression: Head Size vs Brain Weight')
        plt.legend()
        plt.grid(True)

        plot_path = '/home/anild/ros2_ws/src/linreg_analysis/brain_weight_plot.png'
        plt.savefig(plot_path)
        self.get_logger().info(f"Prediction plot saved to {plot_path}")

        # Keep node alive for rqt_graph visibility
        self.create_timer(1.0, lambda: None)

def main(args=None):
    rclpy.init(args=args)
    node = BrainWeightNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

