import streamlit as st
import cv2
import os
from PIL import Image
from edit_picture.edit import Edit
from utils.image_files import *
from streamlit_drawable_canvas import st_canvas
from model.HOG import *
from model.Logistic import *
import numpy as np

st.title("Ứng dụng chỉnh sửa ảnh và nhận diện khuôn mặt")
# hiển thị ảnh thù thư mục "db"
db_images_dir = "db"
st.sidebar.header("Ảnh gợi ý")
db_images = os.listdir(db_images_dir)  
db_images = [img for img in db_images if img.endswith(('jpg', 'jpeg', 'png'))]
db_image_paths = [os.path.join(db_images_dir, img) for img in db_images]
selected_db_image = st.sidebar.selectbox(
    "Chọn ảnh gợi ý:", 
    options=["Không chọn ảnh gợi ý"] + db_images,  
    format_func=lambda x: f"Ảnh {x}" if x != "Không chọn ảnh gợi ý" else "Không chọn ảnh gợi ý"
)

# Biến xác định chọn ảnh gợi ý
use_db_image = selected_db_image != "Không chọn ảnh gợi ý" and selected_db_image != ""

# Khi người dùng chọn ảnh từ gợi ý
if use_db_image:
    selected_image_path = os.path.join(db_images_dir, selected_db_image)
    cv2_image = cv2.imread(selected_image_path)  # Đọc ảnh từ thư mục
    st.sidebar.image(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB), caption="Ảnh gợi ý", use_column_width=True)
    image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)  
    edit = Edit(cv2_image)
    st.image(image_rgb, caption="Ảnh gốc", use_column_width=True)

    option = st.selectbox("Chọn thao tác", ["Chỉnh sửa độ sáng", "Cắt ảnh", "Xoay ảnh", "Làm mờ"])

    #chỉnh sửa độ sáng
    if option == "Chỉnh sửa độ sáng":
        brightness = st.slider("Điều chỉnh độ sáng:", 0.5, 3.0, 1.0)
        image_brightness = edit.brightness_image(brightness)
        st.image(cv2.cvtColor(image_brightness, cv2.COLOR_BGR2RGB), caption="Ảnh chỉnh độ sáng", use_column_width=True)
        st.download_button(
            label="Tải ảnh",
            data=cv2.imencode('.png', cv2.cvtColor(image_brightness, cv2.COLOR_BGR2RGB))[1].tobytes(),
            file_name="image_brightness.png",
            mime="image/png",
        )
    #xoay ảnh    
    elif option == "Xoay ảnh":
        angle = st.slider("Chọn góc xoay (độ):", 0, 270, 0, step=90)
        rotated_image = edit.rotate_image(angle)
        st.image(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB), caption=f"Ảnh xoay {angle}°", use_column_width=True)
        st.download_button(
            label="Tải ảnh",
            data=cv2.imencode('.png', cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))[1].tobytes(),
            file_name="rotated_image.png",
            mime="image/png",
        )
    #làm mờ    
    elif option == "Làm mờ":
        ksize = st.slider("Chọn kích thước kernel:", 3, 11, 5, step=2)
        blurred_image = edit.blur_image(ksize)
        st.image(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB), caption="Ảnh làm mờ", use_column_width=True)
        st.download_button(
            label="Tải ảnh",
            data=cv2.imencode('.png', cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))[1].tobytes(),
            file_name="blurred_image.png",
            mime="image/png",
        )

    else:
        st.write("Kéo để chọn vùng muốn cắt:")

    # Mở ảnh gợi ý
        image_pil = Image.open(selected_image_path)
        canvas_width, canvas_height = adjust_canvas_size(image_pil)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=image_pil,
            update_streamlit=True,
            height=canvas_height,  # Sử dụng kích thước đã điều chỉnh
            width=canvas_width,
            drawing_mode="rect",
            key="canvas_image_adjusted",
        )

        # Xử lý vùng được chọn
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            # Lấy tọa độ trên canvas
            left = int(objects[-1]["left"])
            top = int(objects[-1]["top"])
            width = int(objects[-1]["width"])
            height = int(objects[-1]["height"])

            # Chuyển đổi tọa độ về kích thước gốc
            scale_x = image_pil.width / canvas_width
            scale_y = image_pil.height / canvas_height
            x0 = int(left * scale_x)
            y0 = int(top * scale_y)
            x1 = int((left + width) * scale_x)
            y1 = int((top + height) * scale_y)

            # Cắt ảnh chính xác bằng OpenCV
            image_cv = np.array(image_pil)  # Chuyển PIL sang OpenCV
            cropped_image = image_cv[y0:y1, x0:x1]

            # Hiển thị và tải ảnh sau khi cắt
            st.write("Ảnh sau khi cắt:")
            st.image(cropped_image, caption="Ảnh đã cắt", use_column_width="auto")
            cv2.imwrite("db/output.jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

            st.download_button(
                label="Tải ảnh",
                data=cv2.imencode('.jpg', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))[1].tobytes(),
                file_name="cropped_image.jpg",
                mime="image/jpeg",
            )
        if st.button("Dự đoán ảnh đã cắt"):
            output_path = os.path.join("db", "output.jpg")

            if os.path.exists(output_path):  # Kiểm tra xem ảnh đã tồn tại chưa
                gender = GenderClassifier()
                cv2_image = cv2.imread(output_path)  # Đọc ảnh đã cắt
                features = gender.extract_hog_features(gender.prepocess_image(cv2_image))
                
                pred = PredictModel()
                pred.load_weights_bias("weights")
                
                result = pred.predict(features)  # Dự đoán nhãn (1: Nữ, 0: Nam)
                prlt = pred.pre_probality(features)  # Xác suất dự đoán "Nữ"
                
                prlt_nam = (1 - prlt) * 100  # Xác suất "Nam" (phần trăm)
                prlt_nu = prlt * 100  # Xác suất "Nữ" (phần trăm)

                # Hiển thị kết quả
                st.write(f"Kết quả dự đoán: {'Nam' if result == 0 else 'Nữ'}")
                st.write(f"Xác suất: **{prlt_nam:.2f}% là Nam**, **{prlt_nu:.2f}% là Nữ**")
            else:
                st.warning("Không tìm thấy ảnh đầu ra. Vui lòng thực hiện cắt ảnh trước.")        

# Tải ảnh lên nếu không chọn ảnh gợi ý
else:
    upload_image = st.file_uploader("Tải ảnh lên:", type=["jpg", "png", "jpeg"])
    
    if upload_image is not None:
        cv2_image = open_image(upload_image)
        image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)  # Chuyển ảnh sang RGB cho Streamlit
        edit = Edit(cv2_image)
        st.image(image_rgb, caption="Ảnh gốc", use_column_width=True)

        option = st.selectbox("Chọn thao tác", ["Chỉnh sửa độ sáng", "Cắt ảnh", "Xoay ảnh", "Làm mờ"])

        if option == "Chỉnh sửa độ sáng":
            brightness = st.slider("Điều chỉnh độ sáng:", 0.5, 3.0, 1.0)
            image_brightness = edit.brightness_image(brightness)
            st.image(cv2.cvtColor(image_brightness, cv2.COLOR_BGR2RGB), caption="Ảnh chỉnh độ sáng", use_column_width=True)
            st.download_button(
                label="Tải ảnh",
                data=cv2.imencode('.png', cv2.cvtColor(image_brightness, cv2.COLOR_BGR2RGB))[1].tobytes(),
                file_name="image_brightness.png",
                mime="image/png",
            )
        elif option == "Xoay ảnh":
            angle = st.slider("Chọn góc xoay (độ):", 0, 270, 0, step=90)
            rotated_image = edit.rotate_image(angle)
            st.image(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB), caption=f"Ảnh xoay {angle}°", use_column_width=True)
            st.download_button(
                label="Tải ảnh",
                data=cv2.imencode('.png', cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))[1].tobytes(),
                file_name="rotated_image.png",
                mime="image/png",
            )
        elif option == "Làm mờ":
            ksize = st.slider("Chọn kích thước kernel:", 3, 11, 5, step=2)
            blurred_image = edit.blur_image(ksize)
            st.image(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB), caption="Ảnh làm mờ", use_column_width=True)
            st.download_button(
                label="Tải ảnh",
                data=cv2.imencode('.png', cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))[1].tobytes(),
                file_name="blurred_image.png",
                mime="image/png",
            )
        else:
              st.write("Kéo để chọn vùng muốn cắt:")

    # Mở ảnh gợi ý
        image_pil = Image.open(upload_image)
        canvas_width, canvas_height = adjust_canvas_size(image_pil)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=image_pil,
            update_streamlit=True,
            height=canvas_height,  # Sử dụng kích thước đã điều chỉnh
            width=canvas_width,
            drawing_mode="rect",
            key="canvas_image_adjusted",
        )

        # Xử lý vùng được chọn
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            # Lấy tọa độ trên canvas
            left = int(objects[-1]["left"])
            top = int(objects[-1]["top"])
            width = int(objects[-1]["width"])
            height = int(objects[-1]["height"])

            # Chuyển đổi tọa độ về kích thước gốc
            scale_x = image_pil.width / canvas_width
            scale_y = image_pil.height / canvas_height
            x0 = int(left * scale_x)
            y0 = int(top * scale_y)
            x1 = int((left + width) * scale_x)
            y1 = int((top + height) * scale_y)

            # Cắt ảnh chính xác bằng OpenCV
            image_cv = np.array(image_pil)  # Chuyển PIL sang OpenCV
            cropped_image = image_cv[y0:y1, x0:x1]

            # Hiển thị và tải ảnh sau khi cắt
            st.write("Ảnh sau khi cắt:")
            st.image(cropped_image, caption="Ảnh đã cắt", use_column_width="auto")
            cv2.imwrite("db/output.jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

            st.download_button(
                label="Tải ảnh",
                data=cv2.imencode('.jpg', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))[1].tobytes(),
                file_name="cropped_image.jpg",
                mime="image/jpeg",
            )
        if st.button("Dự đoán ảnh đã cắt"):
            output_path = os.path.join("db", "output.jpg")

            if os.path.exists(output_path):  # Kiểm tra xem ảnh đã tồn tại chưa
                gender = GenderClassifier()
                cv2_image = cv2.imread(output_path)  # Đọc ảnh đã cắt
                features = gender.extract_hog_features(gender.prepocess_image(cv2_image))
                
                pred = PredictModel()
                pred.load_weights_bias("weights")
                
                result = pred.predict(features)  # Dự đoán nhãn (1: Nữ, 0: Nam)
                prlt = pred.pre_probality(features)  # Xác suất dự đoán "Nữ"
                
                prlt_nam = (1 - prlt) * 100  # Xác suất "Nam" (phần trăm)
                prlt_nu = prlt * 100  # Xác suất "Nữ" (phần trăm)

                # Hiển thị kết quả
                st.write(f"Kết quả dự đoán: {'Nam' if result == 0 else 'Nữ'}")
                st.write(f"Xác suất: **{prlt_nam:.2f}% là Nam**, **{prlt_nu:.2f}% là Nữ**")
            else:
                st.warning("Không tìm thấy ảnh đầu ra. Vui lòng thực hiện cắt ảnh trước.")
