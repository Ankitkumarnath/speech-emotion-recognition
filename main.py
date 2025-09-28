# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import soundfile as sf  # pip install soundfile

# Configure page
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

@st.cache_resource
def load_models_and_artifacts():
    """Load pre-trained models and preprocessing artifacts"""
    try:
        # Load classical models
        rf_model = joblib.load('random_forest_model.pkl')
        lr_model = joblib.load('logistic_regression_model.pkl')
        
        # Load deep learning model
        cnn_model = load_model('cnn_emotion_model.h5')
        
        # Load preprocessing artifacts
        label_encoder = joblib.load('label_encoder.pkl')
        scaler = joblib.load('scaler.pkl')
        
        return rf_model, lr_model, cnn_model, label_encoder, scaler
    
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please train the models first. Error: {e}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Unexpected error loading artifacts: {e}")
        return None, None, None, None, None

class AudioProcessor:
    """Audio processing class for the Streamlit app"""
    
    def __init__(self, target_sr=16000, duration=3.0):
        self.target_sr = target_sr
        self.duration = duration
        self.target_length = int(target_sr * duration)
    
    def preprocess_audio(self, audio_data, sr):
        """Preprocess uploaded audio (numpy array)"""
        try:
            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = librosa.to_mono(audio_data.T)  # soundfile returns (n_samples, channels)
            
            # Resample to target sample rate
            if sr != self.target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            # Trim silence
            audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            # Pad or trim to fixed length
            if len(audio_data) > self.target_length:
                audio_data = audio_data[:self.target_length]
            else:
                audio_data = np.pad(audio_data, (0, self.target_length - len(audio_data)), mode='constant')
            
            return audio_data, sr
        
        except Exception as e:
            st.error(f"Error preprocessing audio: {e}")
            return None, None
    
    def extract_features(self, audio):
        """Extract features from audio"""
        try:
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=13)
            mfcc_features = np.mean(mfcc.T, axis=0)
            
            # Chroma features - use chroma_stft (modern API)
            try:
                chroma = librosa.feature.chroma_stft(y=audio, sr=self.target_sr)
            except Exception:
                # fallback: compute chroma from STFT magnitude
                stft = librosa.stft(audio)
                chroma = librosa.feature.chroma_stft(S=np.abs(stft), sr=self.target_sr)
            chroma_features = np.mean(chroma.T, axis=0)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.target_sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.target_sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0]
            
            spectral_features = np.array([
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
            ])
            
            # Combine all features
            features = np.concatenate([mfcc_features, chroma_features, spectral_features])
            
            # Extract mel-spectrogram for deep learning
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.target_sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return features, mel_spec_db
        
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None, None

def create_audio_visualizations(audio_data, sr, mel_spec_db):
    """Create visualizations for the audio file"""
    
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Waveform', 'Mel-Spectrogram', 'Feature Distribution'),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # Waveform
    time = np.linspace(0, len(audio_data)/sr, len(audio_data))
    fig.add_trace(
        go.Scatter(x=time, y=audio_data, mode='lines', name='Waveform'),
        row=1, col=1
    )
    
    # Mel-spectrogram (using heatmap)
    fig.add_trace(
        go.Heatmap(z=mel_spec_db, name='Mel-Spectrogram'),
        row=2, col=1
    )
    
    # For Feature Distribution we'll just show mel mean across time as a quick bar
    mel_mean = np.mean(mel_spec_db, axis=1)
    top_n = 20
    x = list(range(min(top_n, len(mel_mean))))
    y = mel_mean[:top_n]
    fig.add_trace(
        go.Bar(x=x, y=y, name='Mel mean (first bins)'),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(height=900, showlegend=False)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Time Frames", row=2, col=1)
    fig.update_yaxes(title_text="Mel Frequency Bins", row=2, col=1)
    
    return fig

def predict_emotion(audio_data, sr, models, processor, scaler, label_encoder):
    """Predict emotion from audio data"""
    
    # Preprocess audio
    processed_audio, new_sr = processor.preprocess_audio(audio_data, sr)
    if processed_audio is None:
        return None
    
    # Extract features
    features, mel_spec = processor.extract_features(processed_audio)
    if features is None:
        return None
    
    rf_model, lr_model, cnn_model = models
    
    # Prepare features for classical models
    try:
        features_scaled = scaler.transform(features.reshape(1, -1))
    except Exception as e:
        st.error(f"Scaler transform error: {e}")
        return None
    
    # Prepare spectrogram for CNN
    mel_spec_reshaped = mel_spec.reshape(1, mel_spec.shape[0], mel_spec.shape[1], 1)
    
    # Make predictions
    predictions = {}
    
    # Random Forest
    try:
        rf_pred = rf_model.predict(features_scaled)[0]
        rf_proba = rf_model.predict_proba(features_scaled)[0]
        predictions['Random Forest'] = {
            'prediction': label_encoder.inverse_transform([int(rf_pred)])[0],
            'confidence': float(np.max(rf_proba)),
            'probabilities': dict(zip(label_encoder.classes_, rf_proba))
        }
    except Exception as e:
        predictions['Random Forest'] = {'prediction': 'error', 'confidence': 0.0, 'probabilities': {}}
        st.warning(f"Random Forest predict error: {e}")
    
    # Logistic Regression
    try:
        lr_pred = lr_model.predict(features_scaled)[0]
        lr_proba = lr_model.predict_proba(features_scaled)[0]
        predictions['Logistic Regression'] = {
            'prediction': label_encoder.inverse_transform([int(lr_pred)])[0],
            'confidence': float(np.max(lr_proba)),
            'probabilities': dict(zip(label_encoder.classes_, lr_proba))
        }
    except Exception as e:
        predictions['Logistic Regression'] = {'prediction': 'error', 'confidence': 0.0, 'probabilities': {}}
        st.warning(f"Logistic Regression predict error: {e}")
    
    # CNN
    try:
        cnn_proba = cnn_model.predict(mel_spec_reshaped)[0]
        cnn_pred = int(np.argmax(cnn_proba))
        predictions['CNN'] = {
            'prediction': label_encoder.inverse_transform([cnn_pred])[0],
            'confidence': float(np.max(cnn_proba)),
            'probabilities': dict(zip(label_encoder.classes_, cnn_proba))
        }
    except Exception as e:
        predictions['CNN'] = {'prediction': 'error', 'confidence': 0.0, 'probabilities': {}}
        st.warning(f"CNN predict error: {e}")
    
    return predictions, processed_audio, mel_spec, new_sr

def plot_probability_bars(probabilities, emotion_emojis):
    """Return a simple plotly horizontal bar of probabilities sorted."""
    prob_items = list(probabilities.items())
    prob_items.sort(key=lambda x: x[1], reverse=True)
    emotions = [k.replace('_', ' ').title() for k, v in prob_items]
    probs = [v for k, v in prob_items]
    fig = go.Figure(go.Bar(
        x=probs,
        y=emotions,
        orientation='h',
        text=[f"{p:.1%}" for p in probs],
        textposition='outside'
    ))
    fig.update_layout(xaxis_tickformat='.0%')
    return fig

def main():
    """Main Streamlit app"""
    
    # Title and description
    st.title("🎤 Speech Emotion Recognition")
    st.markdown("""
    Upload an audio file or record your voice to predict the emotion using multiple machine learning models.
    The system uses MFCC, Chroma, and Spectral features with Random Forest, Logistic Regression, and CNN models.
    """)
    
    # Load models
    with st.spinner("Loading models..."):
        rf_model, lr_model, cnn_model, label_encoder, scaler = load_models_and_artifacts()
    
    if rf_model is None or lr_model is None or cnn_model is None:
        st.error("Please train the models first and ensure model files and artifacts exist in the app folder.")
        return
    
    models = (rf_model, lr_model, cnn_model)
    processor = AudioProcessor()
    
    # Sidebar with model information
    st.sidebar.title("Model Information")
    st.sidebar.markdown("""
    **Available Models:**
    - 🌳 Random Forest
    - 📊 Logistic Regression  
    - 🧠 CNN (Deep Learning)
    
    **Emotions Detected (example):**
    - sad, pleasant_surprise, neutral, happy, fear, disgust, angry
    """)
    
    # File upload section
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3', 'ogg', 'flac'],
        help="Upload an audio file in WAV, MP3, OGG, or FLAC format"
    )
    
    st.header("🎙️ Record Audio")
    st.info("Audio recording feature would require additional setup. For now, please upload a file.")
    
    if uploaded_file is not None:
        # Load audio file robustly using soundfile
        try:
            # Seek start
            uploaded_file.seek(0)
            data, sr = sf.read(uploaded_file)  # returns numpy array and sample rate
            # Show audio player (need to reset pointer and feed raw bytes)
            uploaded_file.seek(0)
            st.audio(uploaded_file)
            
            # Basic audio info
            st.subheader("📊 Audio Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                duration = data.shape[0] / sr
                st.metric("Duration", f"{duration:.2f}s")
            with col2:
                st.metric("Sample Rate", f"{sr} Hz")
            with col3:
                st.metric("Samples", data.shape[0])
            
            # Predict emotion
            with st.spinner("Analyzing audio and predicting emotion..."):
                result = predict_emotion(data, sr, models, processor, scaler, label_encoder)
                
                if result is not None:
                    predictions, processed_audio, mel_spec, proc_sr = result
                    
                    # Display predictions
                    st.header("🎯 Emotion Predictions")
                    
                    # Create columns for each model
                    col1, col2, col3 = st.columns(3)
                    
                    columns = [col1, col2, col3]
                    model_names = ['Random Forest', 'Logistic Regression', 'CNN']
                    model_icons = ['🌳', '📊', '🧠']
                    
                    # emoji mapping (update if your labels differ)
                    emotion_emojis = {
                        'sad': '😢',
                        'pleasant_surprise': '😲',
                        'neutral': '😐',
                        'happy': '😊',
                        'fear': '😰',
                        'disgust': '🤢',
                        'angry': '😡'
                    }
                    
                    for i, (model_name, icon) in enumerate(zip(model_names, model_icons)):
                        if model_name in predictions:
                            pred_data = predictions[model_name]
                            with columns[i]:
                                st.subheader(f"{icon} {model_name}")
                                emotion = pred_data.get('prediction', 'unknown')
                                confidence = pred_data.get('confidence', 0.0)
                                emoji = emotion_emojis.get(emotion, '😐')
                                try:
                                    st.markdown(f"### {emoji} {str(emotion).replace('_', ' ').title()}")
                                    st.markdown(f"**Confidence:** {confidence:.2%}")
                                except Exception:
                                    st.markdown(f"### {emoji} {str(emotion)}")
                                    st.markdown(f"**Confidence:** {confidence}")
                                
                                # Probability bar chart (plotly)
                                probabilities = pred_data.get('probabilities', {})
                                if probabilities:
                                    fig_prob = plot_probability_bars(probabilities, emotion_emojis)
                                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Consensus prediction
                    st.header("🤝 Model Consensus")
                    # Get all predictions
                    all_predictions = [pred['prediction'] for pred in predictions.values() if 'prediction' in pred]
                    all_confidences = [pred['confidence'] for pred in predictions.values() if 'confidence' in pred]
                    
                    if len(all_predictions) > 0:
                        prediction_counts = Counter(all_predictions)
                        consensus_emotion = prediction_counts.most_common(1)[0][0]
                        consensus_count = prediction_counts.most_common(1)[0][1]
                        
                        consensus_confidences = [pred['confidence'] for pred in predictions.values() 
                                               if pred.get('prediction') == consensus_emotion]
                        avg_confidence = float(np.mean(consensus_confidences)) if consensus_confidences else 0.0
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            emoji = emotion_emojis.get(consensus_emotion, '😐')
                            st.metric(
                                "Consensus Emotion", 
                                f"{emoji} {consensus_emotion.replace('_', ' ').title()}",
                                f"{consensus_count}/{len(predictions)} models agree"
                            )
                        with col2:
                            st.metric(
                                "Average Confidence", 
                                f"{avg_confidence:.2%}",
                                f"Range: {min(all_confidences):.2%} - {max(all_confidences):.2%}" if all_confidences else ""
                            )
                    else:
                        st.warning("No valid model predictions were produced.")
                        consensus_emotion = "unknown"
                        avg_confidence = 0.0
                    
                    # Visualizations
                    st.header("📈 Audio Analysis Visualizations")
                    viz_fig = create_audio_visualizations(processed_audio, proc_sr, mel_spec)
                    st.plotly_chart(viz_fig, use_container_width=True)
                    
                    # Model comparison chart
                    st.subheader("🔍 Model Comparison")
                    comparison_data = []
                    for model_name, pred_data in predictions.items():
                        comparison_data.append({
                            'Model': model_name,
                            'Predicted Emotion': str(pred_data.get('prediction', 'unknown')).replace('_', ' ').title(),
                            'Confidence': pred_data.get('confidence', 0.0)
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    fig_comparison = px.bar(
                        comparison_df, 
                        x='Model', 
                        y='Confidence',
                        color='Predicted Emotion',
                        title='Model Predictions Comparison',
                        text='Confidence'
                    )
                    fig_comparison.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                    fig_comparison.update_layout(yaxis_tickformat='.0%')
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Feature importance visualization (for Random Forest)
                    try:
                        if hasattr(rf_model, 'feature_importances_'):
                            st.subheader("🔬 Feature Importance Analysis")
                            importances = rf_model.feature_importances_
                            
                            feature_names = (
                                [f'MFCC_{i}' for i in range(13)] + 
                                [f'Chroma_{i}' for i in range(12)] + 
                                ['Spectral_Centroid_Mean', 'Spectral_Centroid_Std',
                                 'Spectral_Rolloff_Mean', 'Spectral_Rolloff_Std',
                                 'Spectral_Bandwidth_Mean', 'Spectral_Bandwidth_Std',
                                 'ZCR_Mean', 'ZCR_Std']
                            )
                            
                            indices = np.argsort(importances)[::-1][:15]
                            top_importances = importances[indices]
                            top_features = [feature_names[i] for i in indices]
                            
                            fig_importance = go.Figure(go.Bar(
                                x=top_importances[::-1],
                                y=[f for f in top_features[::-1]],
                                orientation='h',
                                text=[f"{v:.3f}" for v in top_importances[::-1]],
                                textposition='outside'
                            ))
                            fig_importance.update_layout(title='Top 15 Most Important Features (Random Forest)')
                            st.plotly_chart(fig_importance, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not show feature importance: {e}")
                    
                    # Download predictions
                    st.header("💾 Download Results")
                    
                    results_summary = {
                        'File': uploaded_file.name,
                        'Duration_seconds': duration,
                        'Sample_rate': sr,
                        'Consensus_emotion': consensus_emotion,
                        'Consensus_confidence': avg_confidence,
                        'Models_agreement': f"{consensus_count}/{len(predictions)}" if len(predictions)>0 else "0/0"
                    }
                    
                    for model_name, pred_data in predictions.items():
                        model_key = model_name.replace(' ', '_').lower()
                        results_summary[f'{model_key}_prediction'] = pred_data.get('prediction', '')
                        results_summary[f'{model_key}_confidence'] = pred_data.get('confidence', 0.0)
                    
                    results_df = pd.DataFrame([results_summary])
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Prediction Results (CSV)",
                        data=csv,
                        file_name=f"emotion_prediction_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Failed to process the audio file. Please try again with a different file.")
        
        except Exception as e:
            st.error(f"Error processing audio file: {e}")
            st.exception(e)
    
    # Additional information section
    st.header("ℹ️ About the Models")
    
    with st.expander("Model Details"):
        st.markdown("""
        ### Random Forest 🌳
        - Uses extracted audio features (MFCC, Chroma, Spectral features)
        - Ensemble of 100 decision trees
        - Good interpretability with feature importance
        
        ### Logistic Regression 📊
        - Linear model using the same extracted features
        - Fast prediction and good baseline performance
        - Probabilistic output interpretation
        
        ### Convolutional Neural Network (CNN) 🧠
        - Processes Mel-spectrograms directly
        - Deep learning approach with multiple Conv2D layers
        - Can capture complex patterns in audio spectrograms
        - Uses early stopping to prevent overfitting
        """)
    
    with st.expander("Features Used"):
        st.markdown("""
        ### Audio Features
        - **MFCC (13 features)**: Mel-Frequency Cepstral Coefficients for speech characteristics
        - **Chroma (12 features)**: Pitch class profiles for harmonic content
        - **Spectral Features (8 features)**:
          - Spectral Centroid (mean, std): Brightness of sound
          - Spectral Rolloff (mean, std): Shape of spectrum
          - Spectral Bandwidth (mean, std): Width of spectrum
          - Zero Crossing Rate (mean, std): Rate of signal sign changes
        - **Mel-Spectrograms**: Time-frequency representation for CNN
        """)
    
    with st.expander("Training Details"):
        st.markdown("""
        ### Training Configuration
        - **Data Split**: 80% training, 20% testing
        - **Audio Preprocessing**: 
          - Converted to mono
          - Resampled to 16kHz
          - Trimmed silence
          - Normalized amplitude
          - Fixed length (3 seconds)
        - **Deep Learning**:
          - Early stopping (patience: 10 epochs)
          - Learning rate reduction on plateau
          - Dropout for regularization
          - Batch normalization
        """)
    
if __name__ == "__main__":
    main()
