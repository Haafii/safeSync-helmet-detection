#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <ctime>

class YOLODetector {
private:
    cv::dnn::Net net;
    std::vector<std::string> class_names;
    float conf_threshold;
    float nms_threshold;
    cv::Size input_size;

public:
    YOLODetector(const std::string& model_path, const std::string& class_file,
                 float conf_thresh = 0.25, float nms_thresh = 0.45,
                 const cv::Size& size = cv::Size(640, 640)) 
        : conf_threshold(conf_thresh), nms_threshold(nms_thresh), input_size(size) {
        
        // Load class names
        std::ifstream ifs(class_file);
        std::string line;
        while (std::getline(ifs, line)) {
            class_names.push_back(line);
        }
        
        // Load network
        net = cv::dnn::readNetFromONNX(model_path);
        
        // Use OpenCV's backend for inference
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    void detect(cv::Mat& frame) {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1./255., input_size, cv::Scalar(), true, false);
        
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Post-process
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // First output contains all predictions
        float* data = (float*)outputs[0].data;
        const int dimensions = 84; // Assuming 80 classes + 4 box coordinates
        const int rows = outputs[0].rows;
        
        for (int i = 0; i < rows; i++) {
            float* row = data + i * dimensions;
            
            // Get confidence and class scores
            float* scores = row + 4;
            int class_id = std::max_element(scores, scores + 80) - scores;
            float confidence = scores[class_id];
            
            if (confidence >= conf_threshold) {
                float x = row[0];
                float y = row[1];
                float w = row[2];
                float h = row[3];
                
                int left = int((x - 0.5 * w) * frame.cols);
                int top = int((y - 0.5 * h) * frame.rows);
                int width = int(w * frame.cols);
                int height = int(h * frame.rows);
                
                boxes.push_back(cv::Rect(left, top, width, height));
                class_ids.push_back(class_id);
                confidences.push_back(confidence);
            }
        }

        // Apply NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

        // Draw detections
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            
            // Draw rectangle
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            
            // Draw label
            std::string label = class_names[class_ids[idx]] + ": " + 
                              cv::format("%.2f", confidences[idx]);
            
            int baseline;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                                0.5, 1, &baseline);
            cv::putText(frame, label, cv::Point(box.x, box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        }
    }
};

int main() {
    // Initialize camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // Initialize YOLOv8 detector
    // Note: You'll need to convert your .pt model to ONNX format first
    YOLODetector detector("best.onnx", "classes.txt");

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Could not read frame." << std::endl;
            break;
        }

        // Perform detection
        detector.detect(frame);

        // Display result
        cv::imshow("YOLOv8 Detection", frame);

        // Break loop on 'q' key
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}