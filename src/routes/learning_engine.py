from flask import Blueprint, request, jsonify
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import sqlite3
import os

learning_engine_bp = Blueprint('learning_engine', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AILearningEngine:
    def __init__(self):
        self.feedback_data = []
        self.quality_metrics = {
            'image_generation': {'avg_rating': 0, 'improvement_rate': 0},
            'video_generation': {'avg_rating': 0, 'improvement_rate': 0},
            'game_creation': {'avg_rating': 0, 'improvement_rate': 0},
            'audio_generation': {'avg_rating': 0, 'improvement_rate': 0},
            'chat_assistance': {'avg_rating': 0, 'improvement_rate': 0}
        }
        self.learning_models = {
            'prompt_optimizer': {},
            'style_predictor': {},
            'quality_assessor': {},
            'trend_analyzer': {}
        }
        self.user_profiles = {}
        
    def collect_user_feedback(self, feedback_data):
        """Collect and process user feedback for learning"""
        try:
            processed_feedback = {
                'user_id': feedback_data.get('user_id'),
                'service_type': feedback_data.get('service_type'),
                'generation_id': feedback_data.get('generation_id'),
                'rating': feedback_data.get('rating', 0),
                'prompt': feedback_data.get('prompt', ''),
                'output_quality': feedback_data.get('output_quality', 0),
                'engagement_time': feedback_data.get('engagement_time', 0),
                'user_action': feedback_data.get('user_action', ''),  # download, share, regenerate
                'timestamp': datetime.now().isoformat()
            }
            
            self.feedback_data.append(processed_feedback)
            
            # Update user profile
            self._update_user_profile(processed_feedback)
            
            # Update quality metrics
            self._update_quality_metrics(processed_feedback)
            
            return processed_feedback
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {str(e)}")
            raise
    
    def _update_user_profile(self, feedback):
        """Update user preference profile based on feedback"""
        user_id = feedback['user_id']
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'preferences': defaultdict(list),
                'avg_ratings': defaultdict(list),
                'engagement_patterns': [],
                'favorite_styles': [],
                'created_at': datetime.now().isoformat()
            }
        
        profile = self.user_profiles[user_id]
        
        # Update preferences
        service_type = feedback['service_type']
        profile['preferences'][service_type].append({
            'prompt': feedback['prompt'],
            'rating': feedback['rating'],
            'timestamp': feedback['timestamp']
        })
        
        # Update average ratings
        profile['avg_ratings'][service_type].append(feedback['rating'])
        
        # Update engagement patterns
        profile['engagement_patterns'].append({
            'service': service_type,
            'engagement_time': feedback['engagement_time'],
            'action': feedback['user_action'],
            'timestamp': feedback['timestamp']
        })
        
        # Analyze favorite styles (simplified)
        if feedback['rating'] >= 4:
            style_keywords = self._extract_style_keywords(feedback['prompt'])
            profile['favorite_styles'].extend(style_keywords)
    
    def _update_quality_metrics(self, feedback):
        """Update overall quality metrics for each service"""
        service_type = feedback['service_type']
        
        if service_type in self.quality_metrics:
            # Get recent ratings for this service
            recent_ratings = [
                f['rating'] for f in self.feedback_data 
                if f['service_type'] == service_type and 
                datetime.fromisoformat(f['timestamp']) > datetime.now() - timedelta(days=7)
            ]
            
            if recent_ratings:
                self.quality_metrics[service_type]['avg_rating'] = np.mean(recent_ratings)
                
                # Calculate improvement rate (simplified)
                if len(recent_ratings) > 1:
                    recent_avg = np.mean(recent_ratings[-5:])  # Last 5 ratings
                    older_avg = np.mean(recent_ratings[:-5]) if len(recent_ratings) > 5 else recent_avg
                    improvement_rate = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                    self.quality_metrics[service_type]['improvement_rate'] = improvement_rate
    
    def optimize_prompt(self, original_prompt, service_type, user_id=None):
        """Optimize prompts based on learning data"""
        try:
            optimized_prompt = original_prompt
            optimization_suggestions = []
            
            # Analyze successful prompts for this service type
            successful_prompts = [
                f['prompt'] for f in self.feedback_data 
                if f['service_type'] == service_type and f['rating'] >= 4
            ]
            
            if successful_prompts:
                # Extract common successful patterns
                successful_keywords = self._extract_common_keywords(successful_prompts)
                
                # Add successful keywords if not present
                for keyword, frequency in successful_keywords[:3]:
                    if keyword.lower() not in original_prompt.lower():
                        optimization_suggestions.append(f"Consider adding '{keyword}' for better results")
                        optimized_prompt += f", {keyword}"
            
            # User-specific optimization
            if user_id and user_id in self.user_profiles:
                user_profile = self.user_profiles[user_id]
                favorite_styles = user_profile.get('favorite_styles', [])
                
                # Add user's favorite styles
                for style in favorite_styles[:2]:
                    if style not in original_prompt.lower():
                        optimization_suggestions.append(f"Added your preferred style: {style}")
                        optimized_prompt += f", {style} style"
            
            # Service-specific optimizations
            if service_type == 'image_generation':
                if 'high quality' not in original_prompt.lower():
                    optimized_prompt += ", high quality, detailed"
                    optimization_suggestions.append("Added quality enhancers")
                    
            elif service_type == 'video_generation':
                if 'smooth' not in original_prompt.lower():
                    optimized_prompt += ", smooth motion, cinematic"
                    optimization_suggestions.append("Added motion quality enhancers")
                    
            elif service_type == 'game_creation':
                if 'engaging' not in original_prompt.lower():
                    optimized_prompt += ", engaging gameplay, intuitive controls"
                    optimization_suggestions.append("Added engagement enhancers")
            
            return {
                'original_prompt': original_prompt,
                'optimized_prompt': optimized_prompt,
                'suggestions': optimization_suggestions,
                'confidence_score': min(len(successful_prompts) / 10, 1.0)  # Max confidence at 10+ examples
            }
            
        except Exception as e:
            logger.error(f"Error optimizing prompt: {str(e)}")
            return {
                'original_prompt': original_prompt,
                'optimized_prompt': original_prompt,
                'suggestions': [],
                'confidence_score': 0
            }
    
    def predict_quality(self, prompt, service_type):
        """Predict the likely quality of a generation based on prompt"""
        try:
            # Analyze similar prompts and their ratings
            similar_prompts = []
            prompt_keywords = set(prompt.lower().split())
            
            for feedback in self.feedback_data:
                if feedback['service_type'] == service_type:
                    feedback_keywords = set(feedback['prompt'].lower().split())
                    similarity = len(prompt_keywords.intersection(feedback_keywords)) / len(prompt_keywords.union(feedback_keywords))
                    
                    if similarity > 0.3:  # 30% similarity threshold
                        similar_prompts.append({
                            'similarity': similarity,
                            'rating': feedback['rating'],
                            'prompt': feedback['prompt']
                        })
            
            if similar_prompts:
                # Weight ratings by similarity
                weighted_ratings = [p['rating'] * p['similarity'] for p in similar_prompts]
                predicted_quality = np.mean(weighted_ratings)
                confidence = min(len(similar_prompts) / 5, 1.0)  # Max confidence at 5+ similar prompts
            else:
                # Default prediction based on service average
                predicted_quality = self.quality_metrics.get(service_type, {}).get('avg_rating', 3.0)
                confidence = 0.1
            
            return {
                'predicted_rating': round(predicted_quality, 2),
                'confidence': round(confidence, 2),
                'similar_examples': len(similar_prompts),
                'recommendation': self._get_quality_recommendation(predicted_quality)
            }
            
        except Exception as e:
            logger.error(f"Error predicting quality: {str(e)}")
            return {
                'predicted_rating': 3.0,
                'confidence': 0.0,
                'similar_examples': 0,
                'recommendation': 'No prediction available'
            }
    
    def get_personalized_recommendations(self, user_id, service_type):
        """Get personalized recommendations for a user"""
        try:
            if user_id not in self.user_profiles:
                return {
                    'recommendations': ['Try different styles to help us learn your preferences'],
                    'suggested_prompts': [],
                    'confidence': 0.0
                }
            
            profile = self.user_profiles[user_id]
            recommendations = []
            suggested_prompts = []
            
            # Analyze user's rating patterns
            user_ratings = profile['avg_ratings'].get(service_type, [])
            if user_ratings:
                avg_rating = np.mean(user_ratings)
                
                if avg_rating < 3.0:
                    recommendations.append("Try more specific and detailed prompts for better results")
                    recommendations.append("Consider using style keywords like 'photorealistic' or 'artistic'")
                elif avg_rating >= 4.0:
                    recommendations.append("Great! Your prompts are working well. Try exploring new styles")
            
            # Suggest based on favorite styles
            favorite_styles = profile.get('favorite_styles', [])
            if favorite_styles:
                most_common_styles = self._get_most_common(favorite_styles)
                for style, count in most_common_styles[:2]:
                    suggested_prompts.append(f"Create something in {style} style")
            
            # Suggest based on successful prompts from similar users
            similar_users = self._find_similar_users(user_id, service_type)
            for similar_user_id in similar_users[:2]:
                similar_profile = self.user_profiles[similar_user_id]
                successful_prompts = [
                    p['prompt'] for p in similar_profile['preferences'].get(service_type, [])
                    if p['rating'] >= 4
                ]
                if successful_prompts:
                    suggested_prompts.extend(successful_prompts[:1])
            
            confidence = min(len(user_ratings) / 10, 1.0)
            
            return {
                'recommendations': recommendations,
                'suggested_prompts': suggested_prompts[:3],
                'confidence': round(confidence, 2),
                'user_stats': {
                    'total_generations': len(user_ratings),
                    'average_rating': round(np.mean(user_ratings), 2) if user_ratings else 0,
                    'favorite_styles': most_common_styles[:3] if favorite_styles else []
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return {
                'recommendations': ['Error generating recommendations'],
                'suggested_prompts': [],
                'confidence': 0.0
            }
    
    def analyze_improvement_trends(self):
        """Analyze improvement trends across all services"""
        try:
            trends = {}
            
            for service_type in self.quality_metrics:
                service_feedback = [
                    f for f in self.feedback_data 
                    if f['service_type'] == service_type
                ]
                
                if len(service_feedback) >= 5:
                    # Sort by timestamp
                    service_feedback.sort(key=lambda x: x['timestamp'])
                    
                    # Calculate trend over time
                    ratings = [f['rating'] for f in service_feedback]
                    timestamps = [datetime.fromisoformat(f['timestamp']) for f in service_feedback]
                    
                    # Simple linear trend calculation
                    if len(ratings) > 1:
                        x = np.arange(len(ratings))
                        trend_slope = np.polyfit(x, ratings, 1)[0]
                        
                        trends[service_type] = {
                            'current_avg': round(np.mean(ratings[-5:]), 2),  # Last 5 ratings
                            'overall_avg': round(np.mean(ratings), 2),
                            'trend_slope': round(trend_slope, 4),
                            'trend_direction': 'improving' if trend_slope > 0 else 'declining' if trend_slope < 0 else 'stable',
                            'total_feedback': len(ratings),
                            'latest_rating': ratings[-1]
                        }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return {}
    
    def _extract_style_keywords(self, prompt):
        """Extract style-related keywords from prompt"""
        style_keywords = [
            'photorealistic', 'artistic', 'abstract', 'minimalist', 'vintage',
            'modern', 'classical', 'digital', 'watercolor', 'oil painting',
            'sketch', 'cartoon', 'anime', 'realistic', 'fantasy', 'sci-fi'
        ]
        
        found_keywords = []
        prompt_lower = prompt.lower()
        
        for keyword in style_keywords:
            if keyword in prompt_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_common_keywords(self, prompts):
        """Extract common keywords from a list of prompts"""
        word_count = defaultdict(int)
        
        for prompt in prompts:
            words = prompt.lower().split()
            for word in words:
                if len(word) > 3 and word not in ['the', 'and', 'for', 'with', 'this', 'that']:
                    word_count[word] += 1
        
        return sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    def _get_most_common(self, items):
        """Get most common items from a list"""
        item_count = defaultdict(int)
        for item in items:
            item_count[item] += 1
        return sorted(item_count.items(), key=lambda x: x[1], reverse=True)
    
    def _find_similar_users(self, user_id, service_type):
        """Find users with similar preferences"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        user_styles = set(user_profile.get('favorite_styles', []))
        
        similar_users = []
        
        for other_user_id, other_profile in self.user_profiles.items():
            if other_user_id != user_id:
                other_styles = set(other_profile.get('favorite_styles', []))
                similarity = len(user_styles.intersection(other_styles)) / len(user_styles.union(other_styles)) if user_styles.union(other_styles) else 0
                
                if similarity > 0.3:  # 30% similarity threshold
                    similar_users.append((other_user_id, similarity))
        
        # Sort by similarity and return user IDs
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return [user_id for user_id, similarity in similar_users]
    
    def _get_quality_recommendation(self, predicted_rating):
        """Get recommendation based on predicted quality"""
        if predicted_rating >= 4.0:
            return "This prompt is likely to produce high-quality results!"
        elif predicted_rating >= 3.0:
            return "This prompt should produce decent results. Consider adding more details."
        else:
            return "This prompt might need improvement. Try being more specific or adding style keywords."

# Initialize learning engine
learning_engine = AILearningEngine()

@learning_engine_bp.route('/collect-feedback', methods=['POST'])
def collect_feedback():
    """Collect user feedback for learning"""
    try:
        feedback_data = request.get_json()
        
        if not feedback_data:
            return jsonify({
                'success': False,
                'error': 'No feedback data provided'
            }), 400
        
        processed_feedback = learning_engine.collect_user_feedback(feedback_data)
        
        return jsonify({
            'success': True,
            'message': 'Feedback collected successfully',
            'feedback_id': processed_feedback.get('generation_id'),
            'timestamp': processed_feedback.get('timestamp')
        })
        
    except Exception as e:
        logger.error(f"Error collecting feedback: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@learning_engine_bp.route('/optimize-prompt', methods=['POST'])
def optimize_prompt():
    """Optimize a prompt based on learning data"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                'success': False,
                'error': 'Prompt is required'
            }), 400
        
        original_prompt = data['prompt']
        service_type = data.get('service_type', 'image_generation')
        user_id = data.get('user_id')
        
        optimization_result = learning_engine.optimize_prompt(original_prompt, service_type, user_id)
        
        return jsonify({
            'success': True,
            'message': 'Prompt optimized successfully',
            'optimization': optimization_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error optimizing prompt: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@learning_engine_bp.route('/predict-quality', methods=['POST'])
def predict_quality():
    """Predict the quality of a generation based on prompt"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                'success': False,
                'error': 'Prompt is required'
            }), 400
        
        prompt = data['prompt']
        service_type = data.get('service_type', 'image_generation')
        
        prediction = learning_engine.predict_quality(prompt, service_type)
        
        return jsonify({
            'success': True,
            'message': 'Quality prediction generated',
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error predicting quality: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@learning_engine_bp.route('/get-recommendations/<user_id>/<service_type>', methods=['GET'])
def get_recommendations(user_id, service_type):
    """Get personalized recommendations for a user"""
    try:
        recommendations = learning_engine.get_personalized_recommendations(user_id, service_type)
        
        return jsonify({
            'success': True,
            'message': 'Recommendations generated successfully',
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@learning_engine_bp.route('/analyze-trends', methods=['GET'])
def analyze_trends():
    """Analyze improvement trends across all services"""
    try:
        trends = learning_engine.analyze_improvement_trends()
        
        return jsonify({
            'success': True,
            'message': 'Trend analysis completed',
            'trends': trends,
            'summary': {
                'total_feedback': len(learning_engine.feedback_data),
                'active_users': len(learning_engine.user_profiles),
                'services_analyzed': len(trends)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error analyzing trends: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@learning_engine_bp.route('/get-quality-metrics', methods=['GET'])
def get_quality_metrics():
    """Get current quality metrics for all services"""
    try:
        return jsonify({
            'success': True,
            'message': 'Quality metrics retrieved',
            'metrics': learning_engine.quality_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting quality metrics: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@learning_engine_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'AI Learning Engine is operational',
        'status': 'healthy',
        'stats': {
            'total_feedback': len(learning_engine.feedback_data),
            'active_users': len(learning_engine.user_profiles),
            'services_tracked': len(learning_engine.quality_metrics)
        },
        'timestamp': datetime.now().isoformat()
    })

