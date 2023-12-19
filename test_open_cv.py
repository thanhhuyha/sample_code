
import cv2
import numpy as np
import threading
import time

def match_template(thread_id, image, template):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    print(f"Thread {thread_id} finished processing.")

# Create two random images of size 2K x 2K and 4K x 4K
image_2k = np.random.randint(0, 256, (2000, 2000), dtype=np.uint8)
template_4k = np.random.randint(0, 256, (4000, 4000), dtype=np.uint8)


# Create 16 threads for template matching
threads = []

# Start time measurement
start_time = time.time()

for i in range(16):
    thread = threading.Thread(target=match_template, args=(i, template_4k, image_2k))
    threads.append(thread)

# Start the threads
for thread in threads:
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

# End time measurement
end_time = time.time()

# Calculate total time elapsed
total_time = end_time - start_time
print(f"All threads have completed. Total time elapsed: {total_time:.2f} seconds.")

#include <iostream>
#include <vector>
#include <thread>
#include <opencv2/opencv.hpp>

void matchTemplateThread(int thread_id, const cv::Mat& image, const cv::Mat& templ) {
    cv::Mat result;
    cv::matchTemplate(image, templ, result, cv::TM_CCOEFF_NORMED);
    std::cout << "Thread " << thread_id << " finished processing." << std::endl;
}

int main() {
    // Create a 4K x 4K image and a 2K x 2K template with random values in grayscale
    cv::Mat image = cv::Mat::zeros(4000, 4000, CV_8UC1);
    cv::randu(image, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Mat templateImg = cv::Mat::zeros(2000, 2000, CV_8UC1);
    cv::randu(templateImg, cv::Scalar::all(0), cv::Scalar::all(255));

    // Create 16 threads
    std::vector<std::thread> threads;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 16; ++i) {
        threads.emplace_back(matchTemplateThread, i, std::ref(image), std::ref(templateImg));
    }

    // Join the threads with the main thread
    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "All threads have completed. Total time elapsed: " << elapsed.count() << " seconds." << std::endl;

    return 0;
}
import cv2
import numpy as np
import time


im1 = np.random.rand(4000,4000).astype(np.float32)
im2 = np.random.rand(2000,2000).astype(np.float32)
start_time = time.time()

for i in range (16):
    result = cv2.matchTemplate(im1, im2, cv2.TM_CCOEFF_NORMED)

end_time = time.time()
processing_time = end_time - start_time

print(f"Template matching took {processing_time:.4f} seconds.")
g++ -std=c++11 -O3 -march=native -flto -ffast-math -o template_matching template_matching.cpp `pkg-config --cflags --libs opencv4` -lpthread
