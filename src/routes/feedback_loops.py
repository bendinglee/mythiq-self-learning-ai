from flask import Blueprint, request, jsonify
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import threading
import time
import requests

feedback_loops_bp = Blueprint('feedback_loops', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackLoopSystem:
    def __init__(self):
        self.active_loops = {}
        self.improvement_algorithms = {}
        self.quality_history = defaultdict(list)
        self.auto_improvement_enabled = True
        self.improvement_threshold = 0.1  # Minimum improvement to trigger changes
        self.feedback_buffer = defaultdict(list)
        self.learning_rate = 0.01
        
        # Initialize improvement algorithms
        self._initialize_improvement_algorithms()
        
        # Start background improvement process
        self._start_background_improvement()
    
    def _initialize_improvement_algorithms(self):
        """Initialize different improvement algorithms for each service"""
        
        self.improvement_algorithms = {
            'image_generation': {
                'prompt_enhancement': self._improve_image_prompts,
                'quality_optimization': self._optimize_image_quality,
                'style_adaptation': self._adapt_image_styles
            },
            'video_generation': {
                'motion_improvement': self._improve_video_motion,
                'scene_optimization': self._optimize_video_scenes,
                'temporal_consistency': self._improve_temporal_consistency
            },
            'game_creation': {
                'engagement_optimization': self._optimize_game_engagement,
                'difficulty_balancing': self._balance_game_difficulty,
                'mechanic_improvement': self._improve_game_mechanics
            },
            'audio_generation': {
                'quality_enhancement': self._enhance_audio_quality,
                'style_matching': self._match_audio_styles,
                'composition_improvement': self._improve_audio_composition
            },
            'chat_assistance': {
                'response_optimization': self._optimize_chat_responses,
                'context_improvement': self._improve_context_understanding,
                'personality_adaptation': self._adapt_chat_personality
            }
        }
    
    def create_feedback_loop(self, service_type, loop_config):
        """Create a new feedback loop for a service"""
        try:
            loop_id = f"{service_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            feedback_loop = {
                'id': loop_id,
                'service_type': service_type,
                'config': loop_config,
                'status': 'active',
                'created_at': datetime.now().isoformat(),
                'metrics': {
                    'feedback_count': 0,
                    'improvement_cycles': 0,
                    'quality_improvement': 0.0,
                    'last_improvement': None
                },
                'improvement_history': []
            }
            
            self.active_loops[loop_id] = feedback_loop
            
            logger.info(f"Created feedback loop {loop_id} for {service_type}")
            return feedback_loop
            
        except Exception as e:
            logger.error(f"Error creating feedback loop: {str(e)}")
            raise
    
    def process_feedback(self, feedback_data):
        """Process incoming feedback and trigger improvements"""
        try:
            service_type = feedback_data.get('service_type')
            
            if not service_type:
                raise ValueError("Service type is required")
            
            # Add to feedback buffer
            self.feedback_buffer[service_type].append({
                'feedback': feedback_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update quality history
            quality_score = self._calculate_quality_score(feedback_data)
            self.quality_history[service_type].append({
                'score': quality_score,
                'timestamp': datetime.now().isoformat(),
                'feedback_id': feedback_data.get('generation_id')
            })
            
            # Check if improvement is needed
            if self._should_trigger_improvement(service_type):
                self._trigger_improvement_cycle(service_type)
            
            return {
                'processed': True,
                'quality_score': quality_score,
                'improvement_triggered': self._should_trigger_improvement(service_type)
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            raise
    
    def _calculate_quality_score(self, feedback_data):
        """Calculate a comprehensive quality score from feedback"""
        try:
            # Base rating (1-5 scale)
            rating = feedback_data.get('rating', 3.0)
            
            # Engagement factor
            engagement_time = feedback_data.get('engagement_time', 0)
            engagement_factor = min(engagement_time / 60, 2.0)  # Max 2x boost for 1+ minute engagement
            
            # Action factor
            user_action = feedback_data.get('user_action', '')
            action_factor = {
                'download': 1.5,
                'share': 1.8,
                'regenerate': 0.7,
                'delete': 0.3,
                '': 1.0
            }.get(user_action, 1.0)
            
            # Calculate composite score
            quality_score = (rating * 0.6 + engagement_factor * 0.2 + action_factor * 0.2) / 5.0
            
            return min(max(quality_score, 0.0), 1.0)  # Normalize to 0-1
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 0.5  # Default neutral score
    
    def _should_trigger_improvement(self, service_type):
        """Determine if improvement cycle should be triggered"""
        try:
            # Check if we have enough feedback
            if len(self.feedback_buffer[service_type]) < 5:
                return False
            
            # Check quality trend
            recent_scores = [
                item['score'] for item in self.quality_history[service_type][-10:]
            ]
            
            if len(recent_scores) < 5:
                return False
            
            # Calculate trend
            recent_avg = np.mean(recent_scores[-5:])
            older_avg = np.mean(recent_scores[:-5])
            
            # Trigger if quality is declining or stagnant
            return (older_avg - recent_avg) > self.improvement_threshold or recent_avg < 0.6
            
        except Exception as e:
            logger.error(f"Error checking improvement trigger: {str(e)}")
            return False
    
    def _trigger_improvement_cycle(self, service_type):
        """Trigger an improvement cycle for a service"""
        try:
            logger.info(f"Triggering improvement cycle for {service_type}")
            
            # Get recent feedback
            recent_feedback = self.feedback_buffer[service_type][-10:]
            
            # Apply improvement algorithms
            improvements = {}
            if service_type in self.improvement_algorithms:
                for algorithm_name, algorithm_func in self.improvement_algorithms[service_type].items():
                    try:
                        improvement_result = algorithm_func(recent_feedback)
                        improvements[algorithm_name] = improvement_result
                    except Exception as e:
                        logger.error(f"Error in {algorithm_name}: {str(e)}")
            
            # Record improvement cycle
            improvement_record = {
                'timestamp': datetime.now().isoformat(),
                'service_type': service_type,
                'improvements': improvements,
                'feedback_analyzed': len(recent_feedback),
                'quality_before': np.mean([item['score'] for item in self.quality_history[service_type][-5:]]),
                'status': 'completed'
            }
            
            # Update active loops
            for loop_id, loop_data in self.active_loops.items():
                if loop_data['service_type'] == service_type:
                    loop_data['metrics']['improvement_cycles'] += 1
                    loop_data['metrics']['last_improvement'] = datetime.now().isoformat()
                    loop_data['improvement_history'].append(improvement_record)
            
            # Clear processed feedback
            self.feedback_buffer[service_type] = []
            
            return improvement_record
            
        except Exception as e:
            logger.error(f"Error triggering improvement cycle: {str(e)}")
            raise
    
    # Image Generation Improvements
    def _improve_image_prompts(self, feedback_data):
        """Improve image generation prompts based on feedback"""
        try:
            successful_prompts = [
                f['feedback']['prompt'] for f in feedback_data 
                if f['feedback'].get('rating', 0) >= 4
            ]
            
            failed_prompts = [
                f['feedback']['prompt'] for f in feedback_data 
                if f['feedback'].get('rating', 0) <= 2
            ]
            
            # Analyze successful patterns
            successful_keywords = self._extract_keywords(successful_prompts)
            failed_keywords = self._extract_keywords(failed_prompts)
            
            # Find keywords that correlate with success
            positive_keywords = [
                kw for kw in successful_keywords 
                if kw not in failed_keywords or successful_keywords.count(kw) > failed_keywords.count(kw) * 2
            ]
            
            return {
                'type': 'prompt_enhancement',
                'positive_keywords': positive_keywords[:10],
                'successful_patterns': len(successful_prompts),
                'improvement_suggestions': [
                    f"Include '{kw}' for better results" for kw in positive_keywords[:3]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error improving image prompts: {str(e)}")
            return {'type': 'prompt_enhancement', 'error': str(e)}
    
    def _optimize_image_quality(self, feedback_data):
        """Optimize image quality parameters"""
        try:
            quality_ratings = [f['feedback'].get('rating', 0) for f in feedback_data]
            avg_quality = np.mean(quality_ratings)
            
            # Suggest quality improvements
            improvements = []
            if avg_quality < 3.0:
                improvements.extend([
                    "Increase resolution parameters",
                    "Add quality enhancement keywords",
                    "Improve prompt specificity"
                ])
            elif avg_quality < 4.0:
                improvements.extend([
                    "Fine-tune style consistency",
                    "Optimize color balance",
                    "Enhance detail rendering"
                ])
            
            return {
                'type': 'quality_optimization',
                'current_avg_quality': round(avg_quality, 2),
                'improvements': improvements,
                'target_quality': min(avg_quality + 0.5, 5.0)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing image quality: {str(e)}")
            return {'type': 'quality_optimization', 'error': str(e)}
    
    def _adapt_image_styles(self, feedback_data):
        """Adapt image styles based on user preferences"""
        try:
            style_preferences = {}
            
            for f in feedback_data:
                prompt = f['feedback'].get('prompt', '').lower()
                rating = f['feedback'].get('rating', 0)
                
                # Extract style keywords
                style_keywords = [
                    'photorealistic', 'artistic', 'abstract', 'minimalist',
                    'vintage', 'modern', 'classical', 'digital'
                ]
                
                for style in style_keywords:
                    if style in prompt:
                        if style not in style_preferences:
                            style_preferences[style] = []
                        style_preferences[style].append(rating)
            
            # Calculate average ratings for each style
            style_ratings = {
                style: np.mean(ratings) 
                for style, ratings in style_preferences.items()
            }
            
            # Sort by preference
            preferred_styles = sorted(style_ratings.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'type': 'style_adaptation',
                'preferred_styles': preferred_styles[:5],
                'style_analysis': style_ratings,
                'recommendations': [
                    f"Promote {style} style (avg rating: {rating:.1f})" 
                    for style, rating in preferred_styles[:3]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error adapting image styles: {str(e)}")
            return {'type': 'style_adaptation', 'error': str(e)}
    
    # Video Generation Improvements
    def _improve_video_motion(self, feedback_data):
        """Improve video motion quality"""
        try:
            motion_feedback = [
                f for f in feedback_data 
                if 'motion' in f['feedback'].get('prompt', '').lower() or 
                   'movement' in f['feedback'].get('prompt', '').lower()
            ]
            
            if motion_feedback:
                avg_motion_rating = np.mean([f['feedback'].get('rating', 0) for f in motion_feedback])
                
                improvements = []
                if avg_motion_rating < 3.5:
                    improvements.extend([
                        "Enhance motion smoothness algorithms",
                        "Improve frame interpolation",
                        "Optimize motion blur effects"
                    ])
                
                return {
                    'type': 'motion_improvement',
                    'motion_rating': round(avg_motion_rating, 2),
                    'improvements': improvements,
                    'motion_samples': len(motion_feedback)
                }
            
            return {
                'type': 'motion_improvement',
                'message': 'Insufficient motion-related feedback'
            }
            
        except Exception as e:
            logger.error(f"Error improving video motion: {str(e)}")
            return {'type': 'motion_improvement', 'error': str(e)}
    
    def _optimize_video_scenes(self, feedback_data):
        """Optimize video scene composition"""
        try:
            scene_ratings = [f['feedback'].get('rating', 0) for f in feedback_data]
            avg_scene_quality = np.mean(scene_ratings)
            
            improvements = []
            if avg_scene_quality < 3.5:
                improvements.extend([
                    "Improve scene composition rules",
                    "Enhance lighting consistency",
                    "Optimize camera movement"
                ])
            
            return {
                'type': 'scene_optimization',
                'scene_quality': round(avg_scene_quality, 2),
                'improvements': improvements
            }
            
        except Exception as e:
            logger.error(f"Error optimizing video scenes: {str(e)}")
            return {'type': 'scene_optimization', 'error': str(e)}
    
    def _improve_temporal_consistency(self, feedback_data):
        """Improve temporal consistency in videos"""
        try:
            consistency_issues = [
                f for f in feedback_data 
                if f['feedback'].get('rating', 0) < 3 and 
                   'flicker' in f['feedback'].get('prompt', '').lower()
            ]
            
            return {
                'type': 'temporal_consistency',
                'consistency_issues': len(consistency_issues),
                'improvements': [
                    "Enhance frame-to-frame consistency",
                    "Reduce temporal artifacts",
                    "Improve object tracking"
                ] if consistency_issues else ["Temporal consistency is good"]
            }
            
        except Exception as e:
            logger.error(f"Error improving temporal consistency: {str(e)}")
            return {'type': 'temporal_consistency', 'error': str(e)}
    
    # Game Creation Improvements
    def _optimize_game_engagement(self, feedback_data):
        """Optimize game engagement based on user behavior"""
        try:
            engagement_scores = [
                f['feedback'].get('engagement_time', 0) for f in feedback_data
            ]
            avg_engagement = np.mean(engagement_scores) if engagement_scores else 0
            
            improvements = []
            if avg_engagement < 60:  # Less than 1 minute
                improvements.extend([
                    "Improve initial game appeal",
                    "Add more engaging mechanics",
                    "Enhance visual feedback"
                ])
            elif avg_engagement < 300:  # Less than 5 minutes
                improvements.extend([
                    "Add progression systems",
                    "Improve difficulty curve",
                    "Add achievement systems"
                ])
            
            return {
                'type': 'engagement_optimization',
                'avg_engagement_time': round(avg_engagement, 1),
                'improvements': improvements
            }
            
        except Exception as e:
            logger.error(f"Error optimizing game engagement: {str(e)}")
            return {'type': 'engagement_optimization', 'error': str(e)}
    
    def _balance_game_difficulty(self, feedback_data):
        """Balance game difficulty based on user feedback"""
        try:
            difficulty_feedback = [
                f for f in feedback_data 
                if 'difficult' in f['feedback'].get('prompt', '').lower() or 
                   'easy' in f['feedback'].get('prompt', '').lower()
            ]
            
            if difficulty_feedback:
                difficulty_ratings = [f['feedback'].get('rating', 0) for f in difficulty_feedback]
                avg_difficulty_rating = np.mean(difficulty_ratings)
                
                improvements = []
                if avg_difficulty_rating < 3.0:
                    improvements.extend([
                        "Adjust difficulty curve",
                        "Add difficulty options",
                        "Improve tutorial systems"
                    ])
                
                return {
                    'type': 'difficulty_balancing',
                    'difficulty_rating': round(avg_difficulty_rating, 2),
                    'improvements': improvements
                }
            
            return {
                'type': 'difficulty_balancing',
                'message': 'Insufficient difficulty feedback'
            }
            
        except Exception as e:
            logger.error(f"Error balancing game difficulty: {str(e)}")
            return {'type': 'difficulty_balancing', 'error': str(e)}
    
    def _improve_game_mechanics(self, feedback_data):
        """Improve game mechanics based on feedback"""
        try:
            mechanic_ratings = [f['feedback'].get('rating', 0) for f in feedback_data]
            avg_mechanic_rating = np.mean(mechanic_ratings)
            
            improvements = []
            if avg_mechanic_rating < 3.5:
                improvements.extend([
                    "Enhance core game mechanics",
                    "Improve control responsiveness",
                    "Add more interactive elements"
                ])
            
            return {
                'type': 'mechanic_improvement',
                'mechanic_rating': round(avg_mechanic_rating, 2),
                'improvements': improvements
            }
            
        except Exception as e:
            logger.error(f"Error improving game mechanics: {str(e)}")
            return {'type': 'mechanic_improvement', 'error': str(e)}
    
    # Audio Generation Improvements
    def _enhance_audio_quality(self, feedback_data):
        """Enhance audio quality based on feedback"""
        try:
            audio_ratings = [f['feedback'].get('rating', 0) for f in feedback_data]
            avg_audio_quality = np.mean(audio_ratings)
            
            improvements = []
            if avg_audio_quality < 3.5:
                improvements.extend([
                    "Improve audio clarity",
                    "Enhance frequency response",
                    "Reduce audio artifacts"
                ])
            
            return {
                'type': 'quality_enhancement',
                'audio_quality': round(avg_audio_quality, 2),
                'improvements': improvements
            }
            
        except Exception as e:
            logger.error(f"Error enhancing audio quality: {str(e)}")
            return {'type': 'quality_enhancement', 'error': str(e)}
    
    def _match_audio_styles(self, feedback_data):
        """Match audio styles to user preferences"""
        try:
            style_analysis = {}
            
            for f in feedback_data:
                prompt = f['feedback'].get('prompt', '').lower()
                rating = f['feedback'].get('rating', 0)
                
                # Extract audio style keywords
                audio_styles = ['electronic', 'classical', 'rock', 'jazz', 'ambient']
                
                for style in audio_styles:
                    if style in prompt:
                        if style not in style_analysis:
                            style_analysis[style] = []
                        style_analysis[style].append(rating)
            
            # Calculate preferences
            style_preferences = {
                style: np.mean(ratings) 
                for style, ratings in style_analysis.items()
            }
            
            return {
                'type': 'style_matching',
                'style_preferences': style_preferences,
                'recommendations': [
                    f"Enhance {style} generation (rating: {rating:.1f})" 
                    for style, rating in sorted(style_preferences.items(), key=lambda x: x[1], reverse=True)[:3]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error matching audio styles: {str(e)}")
            return {'type': 'style_matching', 'error': str(e)}
    
    def _improve_audio_composition(self, feedback_data):
        """Improve audio composition algorithms"""
        try:
            composition_ratings = [f['feedback'].get('rating', 0) for f in feedback_data]
            avg_composition_rating = np.mean(composition_ratings)
            
            improvements = []
            if avg_composition_rating < 3.5:
                improvements.extend([
                    "Enhance harmonic progression",
                    "Improve rhythm patterns",
                    "Add more musical variety"
                ])
            
            return {
                'type': 'composition_improvement',
                'composition_rating': round(avg_composition_rating, 2),
                'improvements': improvements
            }
            
        except Exception as e:
            logger.error(f"Error improving audio composition: {str(e)}")
            return {'type': 'composition_improvement', 'error': str(e)}
    
    # Chat Assistance Improvements
    def _optimize_chat_responses(self, feedback_data):
        """Optimize chat response quality"""
        try:
            response_ratings = [f['feedback'].get('rating', 0) for f in feedback_data]
            avg_response_quality = np.mean(response_ratings)
            
            improvements = []
            if avg_response_quality < 3.5:
                improvements.extend([
                    "Improve response relevance",
                    "Enhance answer accuracy",
                    "Add more helpful details"
                ])
            
            return {
                'type': 'response_optimization',
                'response_quality': round(avg_response_quality, 2),
                'improvements': improvements
            }
            
        except Exception as e:
            logger.error(f"Error optimizing chat responses: {str(e)}")
            return {'type': 'response_optimization', 'error': str(e)}
    
    def _improve_context_understanding(self, feedback_data):
        """Improve context understanding in chat"""
        try:
            context_issues = [
                f for f in feedback_data 
                if f['feedback'].get('rating', 0) < 3
            ]
            
            return {
                'type': 'context_improvement',
                'context_issues': len(context_issues),
                'improvements': [
                    "Enhance context retention",
                    "Improve conversation flow",
                    "Better topic understanding"
                ] if context_issues else ["Context understanding is good"]
            }
            
        except Exception as e:
            logger.error(f"Error improving context understanding: {str(e)}")
            return {'type': 'context_improvement', 'error': str(e)}
    
    def _adapt_chat_personality(self, feedback_data):
        """Adapt chat personality based on user preferences"""
        try:
            personality_feedback = [
                f for f in feedback_data 
                if 'personality' in f['feedback'].get('prompt', '').lower() or 
                   'tone' in f['feedback'].get('prompt', '').lower()
            ]
            
            if personality_feedback:
                personality_ratings = [f['feedback'].get('rating', 0) for f in personality_feedback]
                avg_personality_rating = np.mean(personality_ratings)
                
                return {
                    'type': 'personality_adaptation',
                    'personality_rating': round(avg_personality_rating, 2),
                    'improvements': [
                        "Adjust conversational tone",
                        "Enhance personality consistency",
                        "Improve emotional intelligence"
                    ] if avg_personality_rating < 3.5 else ["Personality adaptation is working well"]
                }
            
            return {
                'type': 'personality_adaptation',
                'message': 'Insufficient personality feedback'
            }
            
        except Exception as e:
            logger.error(f"Error adapting chat personality: {str(e)}")
            return {'type': 'personality_adaptation', 'error': str(e)}
    
    def _extract_keywords(self, prompts):
        """Extract keywords from a list of prompts"""
        keywords = []
        for prompt in prompts:
            words = prompt.lower().split()
            keywords.extend([
                word for word in words 
                if len(word) > 3 and word not in ['the', 'and', 'for', 'with', 'this', 'that']
            ])
        return keywords
    
    def _start_background_improvement(self):
        """Start background improvement process"""
        def improvement_worker():
            while self.auto_improvement_enabled:
                try:
                    # Check each service for improvement opportunities
                    for service_type in ['image_generation', 'video_generation', 'game_creation', 'audio_generation', 'chat_assistance']:
                        if self._should_trigger_improvement(service_type):
                            self._trigger_improvement_cycle(service_type)
                    
                    # Sleep for 5 minutes before next check
                    time.sleep(300)
                    
                except Exception as e:
                    logger.error(f"Error in background improvement: {str(e)}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        # Start background thread
        improvement_thread = threading.Thread(target=improvement_worker, daemon=True)
        improvement_thread.start()
        logger.info("Background improvement process started")
    
    def get_improvement_status(self):
        """Get current improvement status"""
        try:
            status = {
                'auto_improvement_enabled': self.auto_improvement_enabled,
                'active_loops': len(self.active_loops),
                'services_monitored': list(self.quality_history.keys()),
                'total_feedback_processed': sum(len(feedback) for feedback in self.feedback_buffer.values()),
                'quality_trends': {}
            }
            
            # Calculate quality trends
            for service_type, history in self.quality_history.items():
                if len(history) >= 5:
                    recent_scores = [item['score'] for item in history[-5:]]
                    older_scores = [item['score'] for item in history[:-5]] if len(history) > 5 else recent_scores
                    
                    recent_avg = np.mean(recent_scores)
                    older_avg = np.mean(older_scores)
                    
                    status['quality_trends'][service_type] = {
                        'current_quality': round(recent_avg, 3),
                        'previous_quality': round(older_avg, 3),
                        'improvement': round(recent_avg - older_avg, 3),
                        'trend': 'improving' if recent_avg > older_avg else 'declining' if recent_avg < older_avg else 'stable'
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting improvement status: {str(e)}")
            return {'error': str(e)}

# Initialize feedback loop system
feedback_system = FeedbackLoopSystem()

@feedback_loops_bp.route('/create-loop', methods=['POST'])
def create_feedback_loop():
    """Create a new feedback loop"""
    try:
        data = request.get_json()
        
        if not data or 'service_type' not in data:
            return jsonify({
                'success': False,
                'error': 'Service type is required'
            }), 400
        
        service_type = data['service_type']
        loop_config = data.get('config', {})
        
        feedback_loop = feedback_system.create_feedback_loop(service_type, loop_config)
        
        return jsonify({
            'success': True,
            'message': 'Feedback loop created successfully',
            'loop': feedback_loop,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error creating feedback loop: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@feedback_loops_bp.route('/process-feedback', methods=['POST'])
def process_feedback():
    """Process feedback through the improvement system"""
    try:
        feedback_data = request.get_json()
        
        if not feedback_data:
            return jsonify({
                'success': False,
                'error': 'Feedback data is required'
            }), 400
        
        result = feedback_system.process_feedback(feedback_data)
        
        return jsonify({
            'success': True,
            'message': 'Feedback processed successfully',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@feedback_loops_bp.route('/trigger-improvement/<service_type>', methods=['POST'])
def trigger_improvement(service_type):
    """Manually trigger improvement cycle for a service"""
    try:
        improvement_result = feedback_system._trigger_improvement_cycle(service_type)
        
        return jsonify({
            'success': True,
            'message': f'Improvement cycle triggered for {service_type}',
            'improvement': improvement_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error triggering improvement: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@feedback_loops_bp.route('/get-status', methods=['GET'])
def get_improvement_status():
    """Get current improvement system status"""
    try:
        status = feedback_system.get_improvement_status()
        
        return jsonify({
            'success': True,
            'message': 'Improvement status retrieved',
            'status': status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@feedback_loops_bp.route('/get-loops', methods=['GET'])
def get_active_loops():
    """Get all active feedback loops"""
    try:
        return jsonify({
            'success': True,
            'message': 'Active feedback loops retrieved',
            'loops': feedback_system.active_loops,
            'count': len(feedback_system.active_loops),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting loops: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@feedback_loops_bp.route('/toggle-auto-improvement', methods=['POST'])
def toggle_auto_improvement():
    """Toggle automatic improvement on/off"""
    try:
        data = request.get_json()
        enabled = data.get('enabled', not feedback_system.auto_improvement_enabled)
        
        feedback_system.auto_improvement_enabled = enabled
        
        return jsonify({
            'success': True,
            'message': f'Auto-improvement {"enabled" if enabled else "disabled"}',
            'auto_improvement_enabled': enabled,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error toggling auto-improvement: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@feedback_loops_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'Feedback Loop System is operational',
        'status': 'healthy',
        'active_loops': len(feedback_system.active_loops),
        'auto_improvement': feedback_system.auto_improvement_enabled,
        'timestamp': datetime.now().isoformat()
    })

