# 传统视觉常用起手式
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
from IPython.display import clear_output

# 读取配置文件
class ReadConfig:
    def __init__(self, config_file='config.json'):
        self.config_file = '/home/koji/桌面/assignment1-basics-of-opencv-Es777777/Homework/config.json'
        self.data = self._load_config()
        self._extract_values()
    
    def _load_config(self):
        """加载配置"""
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def _extract_values(self):
        """提取所有数值到属性"""
        # ROI
        self.roi_l = self.data['ROI']['x_left']
        self.roi_d = self.data['ROI']['y_down']
        
        # 红色
        red = self.data['Red_hsv']
        self.red_h_l, self.red_h_u = red['rh_l'], red['rh_u']
        self.red_s_l, self.red_s_u = red['rs_l'], red['rs_u']
        self.red_v_l, self.red_v_u = red['rv_l'], red['rv_u']
        
        # 蓝色
        blue = self.data['Blue_hsv']
        self.blue_h_l, self.blue_h_u = blue['bh_l'], blue['bh_u']
        self.blue_s_l, self.blue_s_u = blue['bs_l'], blue['bs_u']
        self.blue_v_l, self.blue_v_u = blue['bv_l'], blue['bv_u']
        
        # 紫色
        purple = self.data['Purple_hsv']
        self.purple_h_l, self.purple_h_u = purple['ph_l'], purple['ph_u']
        self.purple_s_l, self.purple_s_u = purple['ps_l'], purple['ps_u']
        self.purple_v_l, self.purple_v_u = purple['pv_l'], purple['pv_u']

# 颜色范围封装类
class ColorRange:
    def __init__(self, config):
        self.config = config
        self.ball_ranges = self._setup_ball_ranges()
    
    def _setup_ball_ranges(self):
        """设置球体颜色范围"""
        ball_ranges = {
            'Red Ball': {
                'lower': np.array([self.config.red_h_l, self.config.red_s_l, self.config.red_v_l]),
                'upper': np.array([self.config.red_h_u, self.config.red_s_u, self.config.red_v_u]),
                'display_color': (0, 0, 255)
            },
            'Blue Ball': {
                'lower': np.array([self.config.blue_h_l, self.config.blue_s_l, self.config.blue_v_l]),
                'upper': np.array([self.config.blue_h_u, self.config.blue_s_u, self.config.blue_v_u]),
                'display_color': (255, 0, 0)
            },
            'Purple Ball': {
                'lower': np.array([self.config.purple_h_l, self.config.purple_s_l, self.config.purple_v_l]),
                'upper': np.array([self.config.purple_h_u, self.config.purple_s_u, self.config.purple_v_u]),
                'display_color': (255, 0, 255)
            }
        }
        return ball_ranges
    
    def get_ball_range(self, ball_name):
        """获取指定球体的颜色范围"""
        return self.ball_ranges.get(ball_name, None)
    
    def get_all_ball_names(self):
        """获取所有球体名称"""
        return list(self.ball_ranges.keys())
    
    def create_color_mask(self, hsv_frame, ball_name):
        """为指定球体创建颜色掩膜"""
        ball_range = self.get_ball_range(ball_name)
        if ball_range is None:
            return None
        
        mask = cv2.inRange(hsv_frame, ball_range['lower'], ball_range['upper'])
        return mask
    
    def create_all_masks(self, hsv_frame):
        """为所有球体创建颜色掩膜"""
        masks = {}
        for ball_name in self.get_all_ball_names():
            masks[ball_name] = self.create_color_mask(hsv_frame, ball_name)
        return masks

# 球体检测封装类
class BallDetector:
    def __init__(self, config):
        self.config = config
        self.color_range = ColorRange(config)
        self.min_pixels = 2100  # 检测阈值
    
    def set_roi_region(self, frame):
        """设置ROI区域 - x坐标大于roi_l到右边界的部分"""
        height, width = frame.shape[:2]
    
        # 起始x坐标（roi_l右侧）
        start_x = min(self.config.roi_l + 1, width)  # +1确保x > roi_l
        end_x = width  # 到右边界
    
        # 高度限制
        start_y = 0
        end_y = min(self.config.roi_d, height)
    
        # 截取区域
        roi = frame[start_y:end_y, start_x:end_x]
    
        return roi
    
    def detect_balls(self, frame):
        """检测球体"""
        # 截取ROI区域
        roi = self.set_roi_region(frame)
        
        # 转换到HSV颜色空间
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 创建所有球体的掩膜
        ball_masks = self.color_range.create_all_masks(hsv_roi)
        
        # 统计每种球体的像素点数量
        ball_counts = {}
        for ball_name, mask in ball_masks.items():
            if mask is not None:
                pixel_count = np.sum(mask > 0)
                ball_counts[ball_name] = pixel_count
        
        # 找出像素点最多的球体
        dominant_ball = self._find_dominant_ball(ball_counts)
        
        return dominant_ball, ball_counts, ball_masks, roi
    
    def _find_dominant_ball(self, ball_counts):
        """找出主导球体"""
        if not ball_counts:
            return ('No Ball', 0)
        
        dominant_ball = max(ball_counts.items(), key=lambda x: x[1])
        ball_name, pixel_count = dominant_ball
        
        # 如果像素点数量小于阈值，认为没有检测到有效球体
        if pixel_count < self.min_pixels:
            return ('No Ball', 0)
        
        return dominant_ball
    
    def set_detection_threshold(self, threshold):
        """设置检测阈值"""
        self.min_pixels = threshold
    
    def create_display_frame(self, frame, dominant_ball, ball_counts):
        """创建显示图像"""
        display_frame = frame.copy()
        
        # 添加检测结果文本
        ball_name, ball_count = dominant_ball
        
        if ball_name != 'No Ball':
            ball_color = self.color_range.get_ball_range(ball_name)['display_color']
            text = f"Find: {ball_name} ({ball_count} pixels)"
            cv2.putText(display_frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, ball_color, 2)
        else:
            cv2.putText(display_frame, "Find: No Ball", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame

# 视频处理类
class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.ball_detector = BallDetector(config)
    
    def process_video(self, filename='output1.avi'):
        """处理视频文件"""
        import os
        import warnings
        import sys
        
        warnings.filterwarnings("ignore")
        
        current_path = os.path.abspath(os.path.dirname(__file__))
        filepath = os.path.join(current_path, filename)
        
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            print("无法打开视频文件！")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print("视频文件已成功打开！")
        print(f"总帧数: {total_frames}, FPS: {fps:.2f}")
        print("像素阈值: 2100 pixels")
        print("操作方法:")
        print("Q:后退5帧 W:后退1帧 E:前进1帧 R:前进5帧")
        print("空格:暂停/继续 ESC:退出程序")
        print("-" * 50)
        
        # 判断平台设置延时
        system_platform = sys.platform
        if 'win' in system_platform:
            time_delay = 1
        else:
            time_delay = 40
        
        cv2.namedWindow('Finding Ball', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Finding Ball', 640, 480)
        
        i = 1
        paused = False
        last_print_time = 0
        print_interval = 0.5  # 每0.5秒在终端输出一次
        
        while True:
            current_time = time.time()
            
            if not paused:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    if i >= total_frames:
                        i = 1  # 循环播放
                        continue
                    else:
                        break
                        
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                # 检测球体
                dominant_ball, ball_counts, ball_masks, roi = self.ball_detector.detect_balls(frame)
                
                # 创建显示图像
                display_frame = self.ball_detector.create_display_frame(frame, dominant_ball, ball_counts)
                
                # 显示帧号
                cv2.putText(display_frame, f'Frame_number: {i}/{total_frames}', 
                           (10, display_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 显示图像
                cv2.imshow('Finding Ball', display_frame)
                
                # 终端实时输出
                if current_time - last_print_time > print_interval:
                    self._print_detection_result(i, total_frames, dominant_ball, ball_counts)
                    last_print_time = current_time
                
                i += 1
            
            # 键盘控制
            key = self._handle_keyboard_input(cap, i, paused, total_frames)
            if key == 'exit':
                break
            elif key == 'paused':
                paused = True
            elif key == 'resumed':
                paused = False
            elif isinstance(key, int):
                i = key
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n程序结束")
    
    def _print_detection_result(self, current_frame, total_frames, dominant_ball, ball_counts):
        """打印检测结果"""
        clear_output(wait=True)
        print(f"\rFrame_number: {current_frame}/{total_frames}", end="")
        ball_name, ball_count = dominant_ball
        
        if ball_name != 'No Ball':
            print(f" - 检测到球体: {ball_name} (pixels: {ball_count})")
            for ball_name, count in ball_counts.items():
                status = "✓" if count >= 2100 else "✗"
                print(f"  {ball_name}: {count} pixels {status}")
        else:
            print(" - 未检测到有效球体")
            for ball_name, count in ball_counts.items():
                print(f"  {ball_name}: {count} pixels ✗")
    
    def _handle_keyboard_input(self, cap, current_frame, paused, total_frames):
        """处理键盘输入"""
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        
        if key == ord(' '):  # 空格键暂停/继续
            paused = not paused
            if paused:
                print("\n暂停播放")
            else:
                print("继续播放")
            return 'paused' if paused else 'resumed'
        
        elif key == ord('e') and not paused:  # 前进1帧
            if current_frame < total_frames - 1:
                return current_frame + 1
        
        elif key == ord('r') and not paused:  # 前进5帧
            if current_frame < total_frames - 5:
                return current_frame + 5
        
        elif key == ord('w'):  # 后退1帧
            if current_frame > 1:
                print(f"\n后退到帧 {current_frame - 1}，已暂停")
                return current_frame - 1
        
        elif key == ord('q'):  # 后退5帧
            if current_frame > 5:
                print(f"\n后退到帧 {current_frame - 5}，已暂停")
                return current_frame - 5
        
        elif key == 27:  # ESC退出
            return 'exit'
        
        return current_frame

# 使用示例
if __name__ == "__main__":
    print("=== 球体检测程序启动 ===")
    print("=" * 50)
    
    # 加载配置
    config = ReadConfig()
    
    # 创建视频处理器
    video_processor = VideoProcessor(config)
    
    # 处理视频
    video_processor.process_video('output1.avi')