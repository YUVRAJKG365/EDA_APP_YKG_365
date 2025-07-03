import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature, filters
from skimage.color import rgb2gray, rgb2hsv
import piexif
from PIL import Image
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader, PdfWriter
import gc
import traceback

import tracker
from utils.data_loader import display_file_info, handle_file_upload
from utils.session_state_manager import get_session_manager

def optimize_image_display(img):
    """Optimize image for display while maintaining quality"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def clear_memory_safe():
    """Safely clear memory without causing errors"""
    try:
        gc.collect()
        if 'pdf_images' in st.session_state:
            if st.session_state.pdf_images is not None:
                for img in st.session_state.pdf_images:
                    if img is not None and hasattr(img, 'close'):
                        img.close()
                del st.session_state.pdf_images
        if 'image_data' in st.session_state:
            if st.session_state.image_data is not None and hasattr(st.session_state.image_data, 'close'):
                st.session_state.image_data.close()
            del st.session_state.image_data
    except Exception as e:
        st.warning(f"Memory cleanup warning: {str(e)}")

def try_reconstruct_pdf(file):
    """Safely attempt to reconstruct a PDF"""
    try:
        file.seek(0)
        reader = PdfReader(file)
        writer = PdfWriter()
        
        for page in reader.pages:
            writer.add_page(page)
            
        reconstructed = BytesIO()
        writer.write(reconstructed)
        reconstructed.seek(0)
        return reconstructed
    except Exception as e:
        st.warning(f"PDF reconstruction attempt failed: {str(e)}")
        return None
    finally:
        gc.collect()

def process_pdf_safely(uploaded_fileia, poppler_path):
    """Robust PDF processing with comprehensive error handling"""
    try:
        # First verify PDF integrity
        try:
            pdf_reader = PdfReader(uploaded_fileia)
            page_count = len(pdf_reader.pages)
            if page_count == 0:
                st.error("PDF contains no pages")
                return None
        except Exception as e:
            st.warning(f"Could not read PDF page count: {str(e)}")
            # Try reconstruction
            reconstructed = try_reconstruct_pdf(uploaded_fileia)
            if reconstructed:
                pdf_reader = PdfReader(reconstructed)
                page_count = len(pdf_reader.pages)
            else:
                return None

        # Set reasonable conversion parameters
        dpi = st.slider("Conversion resolution (DPI)", 50, 300, 150, 
                        help="Lower DPI for faster processing of large PDFs")
        batch_size = min(5, page_count)  # Smaller batches for stability
        
        images = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for batch_start in range(0, page_count, batch_size):
                batch_end = min(batch_start + batch_size, page_count)
                current_page = batch_start + 1
                
                # Update progress carefully to avoid overflow
                progress = min(1.0, (batch_end) / page_count)
                progress_bar.progress(progress)
                status_text.text(f"Processing pages {current_page}-{batch_end} of {page_count}...")
                
                uploaded_fileia.seek(0)  # Reset file pointer for each batch
                
                try:
                    batch = convert_from_bytes(
                        uploaded_fileia.read(),
                        poppler_path=poppler_path,
                        first_page=current_page,
                        last_page=batch_end,
                        dpi=dpi,
                        fmt='jpeg',
                        thread_count=1,
                        strict=False
                    )
                    
                    if batch:
                        for img in batch:
                            try:
                                optimized_img = optimize_image_display(img)
                                images.append(optimized_img)
                            except Exception as e:
                                st.warning(f"Skipping a page due to processing error: {str(e)}")
                                continue
                except Exception as e:
                    st.warning(f"Batch {current_page}-{batch_end} failed: {str(e)}")
                    continue
                
                gc.collect()  # Clear memory between batches
            
            if not images:
                st.error("No images could be extracted from PDF")
                return None
                
            return images
            
        finally:
            progress_bar.empty()
            status_text.empty()
            
    except MemoryError:
        st.error("Not enough memory to process this PDF. Try a smaller file or lower DPI.")
        return None
    except Exception as e:
        st.error(f"PDF processing failed: {str(e)}")
        return None
    finally:
        gc.collect()

def render_image_analysis_section():
    """Main image analysis function with robust error handling and session isolation"""
    # Get session manager instance
    session_manager = get_session_manager()
    section = "Image Analysis"
    
    st.markdown("## 🖼️ Advanced Image Analysis")
    st.markdown("Perform comprehensive analysis on uploaded images.")
    
    # File upload with section isolation
    uploaded_file = handle_file_upload(
        section=section,
        file_types=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'pdf'],
        title="Upload an image or PDF file (max 100MB)",
        help_text="Maximum file size: 100MB"
    )
    
    # Display file info if available
    if session_manager.has_data(section, 'file_processed') and session_manager.get_data(section, 'file_processed'):
        display_file_info(section)
        
        # Process uploaded file with comprehensive error handling
        if uploaded_file is not None:
            try:
                # Check file size
                if uploaded_file.size > 100 * 1024 * 1024:  # 100MB limit
                    st.error("File too large (max 100MB)")
                    return
                
                if uploaded_file.type.startswith('image/'):
                    try:
                        with Image.open(uploaded_file) as img:
                            session_manager.set_data(section, 'image_data', optimize_image_display(img.copy()))
                        session_manager.set_data(section, 'pdf_images', None)
                        st.success(f"Successfully loaded image: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
                        clear_memory_safe()
                        return
                
                elif uploaded_file.type == 'application/pdf':
                    poppler_path = r"C:\Users\yuvra\Documents\poppler\poppler-23.11.0\Library\bin"
                    with st.spinner("Processing PDF..."):
                        session_manager.set_data(section, 'pdf_images', process_pdf_safely(uploaded_file, poppler_path))
                    session_manager.set_data(section, 'image_data', None)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                clear_memory_safe()
                return
    
    # Get available images safely
    images = []
    try:
        image_data = session_manager.get_data(section, 'image_data')
        pdf_images = session_manager.get_data(section, 'pdf_images')
        
        if image_data is not None:
            images.append(("Uploaded Image", image_data))
        
        if pdf_images is not None:
            images.extend([
                (f"PDF Page {i+1}", img) 
                for i, img in enumerate(pdf_images)
                if img is not None
            ])
    except Exception as e:
        st.warning(f"Error loading images: {str(e)}")
        clear_memory_safe()
    
    if not images:
        st.info("Please upload an image file or PDF with images to use image analysis features.")
        return

    # Image analysis section
    try:
        # Image selection
        selected_img_name = st.selectbox(
            "Select image to analyze",
            [name for name, img in images],
            key=f"{section}_img_analysis_select"
        )
        selected_img = next(img for name, img in images if name == selected_img_name)
        
        if not isinstance(selected_img, Image.Image):
            st.warning("Selected file is not a valid image.")
            return
            
        # Convert to RGB if needed
        if selected_img.mode != 'RGB':
            selected_img = selected_img.convert('RGB')
            
        # Convert to numpy array with memory check
        try:
            img_array = np.array(selected_img)
        except MemoryError:
            st.error("Not enough memory to process this image")
            clear_memory_safe()
            return
        
        # Display image
        with st.expander("Image Preview", expanded=True):
            st.image(selected_img, caption=selected_img_name, use_container_width=True)

        # Analysis tabs
        tab_labels = [
            "📊 Basic Features",
            "🎨 Color Analysis",
            "🧩 Texture/Shape",
            "🔥 Heatmaps & 3D",
            "🤖 Advanced"
        ]
        
        choice = st.radio(
            "Analysis Type",
            options=tab_labels,
            key=f"{section}_img_analysis_tab_choice",
            horizontal=True
        )
        
        active = tab_labels.index(choice)

        # Branch based on active tab index
        if active == 0:  # Basic Features           
            st.markdown("### Basic Image Features")
            col1, col2 = st.columns(2)
            with col1:
                st.image(selected_img, use_container_width=True)
            with col2:
                st.write(f"**Format:** {selected_img.format if hasattr(selected_img, 'format') else 'Unknown'}")
                st.write(f"**Size:** {selected_img.size} (width x height)")
                st.write(f"**Mode:** {selected_img.mode}")
                st.write(f"**Shape:** {img_array.shape}")
                st.write(f"**Dtype:** {img_array.dtype}")
                st.write(f"**Min Value:** {np.min(img_array)}")
                st.write(f"**Max Value:** {np.max(img_array)}")
                st.write(f"**Mean Value:** {np.mean(img_array):.2f}")
                st.write(f"**Standard Deviation:** {np.std(img_array):.2f}")
                
                if len(img_array.shape) == 3:
                    st.write("**Channel Means:**")
                    st.write(f"- Red: {np.mean(img_array[:,:,0]):.2f}")
                    st.write(f"- Green: {np.mean(img_array[:,:,1]):.2f}")
                    st.write(f"- Blue: {np.mean(img_array[:,:,2]):.2f}")

            # Histogram
            st.markdown("### Histogram")
            try:
                if len(img_array.shape) == 3:  # Color image
                    red = img_array[:,:,0].flatten()
                    green = img_array[:,:,1].flatten()
                    blue = img_array[:,:,2].flatten()
                    
                    df = pd.DataFrame({
                        'Intensity': np.concatenate([red, green, blue]),
                        'Channel': ['Red'] * len(red) + ['Green'] * len(green) + ['Blue'] * len(blue)
                    })
                    
                    fig = px.histogram(
                        df, 
                        x='Intensity', 
                        color='Channel',
                        color_discrete_map={'Red': 'red', 'Green': 'green', 'Blue': 'blue'},
                        nbins=256,
                        opacity=0.7,
                        title="RGB Histogram"
                    )
                    fig.update_layout(barmode='overlay')
                else:  # Grayscale
                    fig = px.histogram(
                        x=img_array.flatten(),
                        labels={'x': 'Pixel Intensity'},
                        title="Intensity Histogram",
                        nbins=256
                    )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate histogram: {e}")
            
            # EXIF data
            try:
                if hasattr(selected_img, '_getexif') and selected_img._getexif():
                    with st.expander("EXIF Metadata"):
                        exif_data = {}
                        for k, v in selected_img._getexif().items():
                            tag_name = piexif.TAGS.get(k, k)
                            if not isinstance(v, (int, float, str, bytes)):
                                try:
                                    v = str(v)
                                except:
                                    v = "Unserializable value"
                            exif_data[tag_name] = v
                        st.json(exif_data)
            except:
                st.warning("Could not read EXIF data")

        elif active == 1:  # Color Analysis
            if len(img_array.shape) == 3:
                st.markdown("### Color Space Analysis")
                color_space = st.selectbox(
                    "Select color space",
                    ["RGB", "HSV", "LAB"],
                    key=f"{section}_color_space"
                )
                
                try:
                    if color_space == "HSV":
                        converted = rgb2hsv(img_array)
                        channel_names = ["Hue", "Saturation", "Value"]
                    elif color_space == "LAB":
                        converted = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                        channel_names = ["Luminance", "A", "B"]
                    else:
                        converted = img_array
                        channel_names = ["Red", "Green", "Blue"]
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    for j in range(3):
                        axes[j].imshow(converted[:,:,j], cmap='gray')
                        axes[j].set_title(f"{channel_names[j]} Channel")
                        axes[j].axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Color space conversion failed: {e}")
                
                # Channel separation
                st.markdown("### Channel Separation")
                cols = st.columns(3)
                for i, channel in enumerate(["Red", "Green", "Blue"]):
                    with cols[i]:
                        channel_img = np.zeros_like(img_array)
                        channel_img[:,:,i] = img_array[:,:,i]
                        st.image(channel_img, caption=channel, use_container_width=True)
                
                # Color clustering
                st.markdown("### Color Clustering")
                if st.checkbox("Perform color clustering", key=f"{section}_color_cluster"):
                    try:
                        n_clusters = st.slider("Number of colors", 2, 10, 3)
                        pixels = img_array.reshape(-1, 3)
                        pixels = np.float32(pixels)

                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                        _, labels, centers = cv2.kmeans(
                            pixels, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
                        )

                        centers = np.uint8(centers)
                        clustered = centers[labels.flatten()]
                        clustered = clustered.reshape(img_array.shape)

                        st.image(clustered, caption=f"Color Clustering ({n_clusters} colors)", use_container_width=True)

                        palette = np.zeros((50, 50*n_clusters, 3), np.uint8)
                        for j in range(n_clusters):
                            palette[:,j*50:(j+1)*50] = centers[j]
                        st.image(palette, caption="Color Palette", use_container_width=True)
                    except Exception as e:
                        st.error(f"Color clustering failed: {e}")
            else:
                st.info("Color analysis requires a color image")

        elif active == 2:  # Texture/Shape
            st.markdown("### Texture and Shape Analysis")
            
            # Texture analysis
            with st.expander("Texture Features"):
                if st.checkbox("Calculate GLCM Texture Features", key=f"{section}_glcm"):
                    try:
                        gray_img = rgb2gray(img_array) if len(img_array.shape) == 3 else img_array
                        gray_img = (gray_img * 255).astype(np.uint8)

                        glcm = feature.graycomatrix(gray_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
                        
                        props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
                        for prop in props:
                            st.write(f"**{prop.capitalize()}:** {feature.graycoprops(glcm, prop)[0,0]:.4f}")
                    except Exception as e:
                        st.error(f"GLCM calculation failed: {e}")
            
            # Shape analysis
            with st.expander("Shape Analysis"):
                if st.checkbox("Detect shapes", key=f"{section}_shape_analysis"):
                    try:
                        threshold = st.slider("Threshold value", 0, 255, 128)
                        gray_img = rgb2gray(img_array) if len(img_array.shape) == 3 else img_array
                        _, binary = cv2.threshold(
                            (gray_img*255).astype(np.uint8),
                            threshold, 255, cv2.THRESH_BINARY
                        )
                        binary = binary.astype(np.uint8)

                        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        st.write(f"**Found {len(contours)} contours**")

                        contour_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                        cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
                        st.image(contour_img, caption="Detected Contours", use_container_width=True)

                        if contours:
                            cnt = contours[0]
                            st.write("**Properties of first contour:**")
                            st.write(f"- Area: {cv2.contourArea(cnt):.2f}")
                            st.write(f"- Perimeter: {cv2.arcLength(cnt, True):.2f}")
                            st.write(f"- Bounding Box: {cv2.boundingRect(cnt)}")
                    except Exception as e:
                        st.error(f"Shape detection failed: {e}")
            
            # Edge detection
            with st.expander("Edge Detection"):
                method = st.selectbox(
                    "Select edge detection method",
                    ["Sobel", "Canny", "Prewitt", "Roberts"],
                    key=f"{section}_edge_method"
                )
                try:
                    gray_img = rgb2gray(img_array) if len(img_array.shape) == 3 else img_array
                    
                    if method == "Sobel":
                        edges = filters.sobel(gray_img)
                    elif method == "Canny":
                        edges = feature.canny(gray_img)
                    elif method == "Prewitt":
                        edges = filters.prewitt(gray_img)
                    else:  # Roberts
                        edges = filters.roberts(gray_img)
                    
                    edge_img = Image.fromarray((edges * 255).astype(np.uint8))
                    st.image(edge_img, caption=f"{method} Edge Detection", use_container_width=True)
                except Exception as e:
                    st.error(f"Edge detection failed: {e}")

        elif active == 3:  # Heatmaps & 3D
            st.markdown("## 🔥 Heatmaps & 3D Visualizations")
            
            # Heatmap
            st.markdown("### Pixel Intensity Heatmap")
            try:
                if len(img_array.shape) == 3:
                    channel = st.selectbox(
                        "Select channel",
                        ["Red", "Green", "Blue"],
                        key=f"{section}_heatmap_channel"
                    )
                    channel_idx = ["Red", "Green", "Blue"].index(channel)
                    heatmap_data = img_array[:,:,channel_idx]
                else:
                    heatmap_data = img_array
                    
                fig = px.imshow(heatmap_data, color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Heatmap generation failed: {e}")
            
            # 3D Color Distribution
            if len(img_array.shape) == 3:
                st.markdown("### 3D Color Distribution")
                try:
                    sample_size = min(5000, img_array.shape[0]*img_array.shape[1])
                    sampled = img_array.reshape(-1, 3)[np.random.choice(
                        img_array.shape[0]*img_array.shape[1],
                        sample_size,
                        replace=False
                    )]
                    
                    fig = px.scatter_3d(
                        x=sampled[:,0], y=sampled[:,1], z=sampled[:,2],
                        color=['rgb('+','.join(map(str, c))+')' for c in sampled],
                        labels={'x': 'Red', 'y': 'Green', 'z': 'Blue'},
                        title="3D Color Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"3D visualization error: {str(e)}")
            else:
                st.info("3D color distribution requires a color image")
                
            # 3D Surface Plot
            st.markdown("### 3D Surface Plot")
            try:
                if len(img_array.shape) == 2:  # Grayscale
                    z_data = img_array
                    fig = go.Figure(data=[go.Surface(z=z_data, colorscale='viridis')])
                    fig.update_layout(
                        title='Intensity Surface',
                        autosize=True,
                        scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Intensity'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Surface plot available for grayscale images only")
            except Exception as e:
                st.error(f"3D surface plot failed: {e}")

        elif active == 4:  # Advanced
            st.markdown("### Advanced Image Analysis")
            
            # Create sub-tabs for Advanced section
            sub_tabs = st.tabs(["Advanced Analysis", "Geospatial"])
            
            with sub_tabs[0]:  # Advanced Analysis
                # Feature extraction
                with st.expander("Feature Extraction"):
                    if st.checkbox("Extract key features", key=f"{section}_feature_extract"):
                        try:
                            gray_img = rgb2gray(img_array) if len(img_array.shape) == 3 else img_array
                            gray_img = (gray_img * 255).astype(np.uint8)

                            sift = cv2.SIFT_create()
                            kp = sift.detect(gray_img, None)
                            img_kp = cv2.drawKeypoints(
                                gray_img,
                                kp,
                                None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                            )
                            st.image(img_kp, caption=f"Detected {len(kp)} keypoints", use_container_width=True)
                        except Exception as e:
                            st.error(f"Feature extraction failed: {e}")
                
                # Image comparison
                if len(images) > 1:
                    with st.expander("Image Similarity"):
                        if st.checkbox("Compare with another image", key=f"{section}_img_compare"):
                            try:
                                other_img_name = st.selectbox(
                                    "Select image to compare with",
                                    [name for name, img in images if name != selected_img_name],
                                    key=f"{section}_img_compare_select"
                                )
                                other_img = next(img for name, img in images if name == other_img_name)
                                other_array = np.array(other_img)

                                if img_array.shape == other_array.shape:
                                    mse = np.mean((img_array - other_array)**2)
                                    st.write(f"**Mean Squared Error:** {mse:.2f}")

                                    from skimage.metrics import structural_similarity as ssim
                                    if len(img_array.shape) == 3:
                                        ssim_score = ssim(img_array, other_array, channel_axis=-1, win_size=3)
                                    else:
                                        ssim_score = ssim(img_array, other_array, win_size=3)
                                    st.write(f"**Structural Similarity:** {ssim_score:.4f}")
                                    
                                    diff = np.abs(img_array.astype(np.float32) - other_array.astype(np.float32))
                                    diff_img = (diff / np.max(diff) * 255).astype(np.uint8)
                                    st.image(diff_img, caption="Difference Image", use_container_width=True)

                                else:
                                    st.warning("Images must be same size for comparison")
                            except Exception as e:
                                st.error(f"Image comparison failed: {e}")
            
            with sub_tabs[1]:  # Geospatial                
                st.markdown("#### Geospatial Visualization")
                st.info("Geospatial analysis for images with geotags.")
                try:
                    if hasattr(selected_img, '_getexif') and selected_img._getexif():
                        exif_dict = piexif.load(selected_img.info['exif'])
                        gps_info = exif_dict.get('GPS')
                        if gps_info:
                            st.write("GPS info found in EXIF:")
                            st.json(gps_info)
                        else:
                            st.warning("No GPS data found in EXIF.")
                    else:
                        st.warning("No EXIF data available for geospatial analysis.")
                except:
                    st.warning("Could not read EXIF data for geospatial analysis.")

    except MemoryError:
        st.error("The operation ran out of memory. Try with a smaller image or fewer features.")
        clear_memory_safe()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.text(traceback.format_exc())  # Show full traceback in debug mode
        clear_memory_safe()
    finally:
        gc.collect()