from flask import Blueprint, request, jsonify
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import requests
import re
import random

adaptive_models_bp = Blueprint('adaptive_models', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveModelSystem:
    def __init__(self):
        self.model_registry = {}
        self.model_performance = defaultdict(list)
        self.prompt_templates = {}
        self.optimization_rules = {}
        self.user_model_preferences = defaultdict(dict)
        self.model_usage_stats = defaultdict(int)
        
        # Initialize model registry
        self._initialize_model_registry()
        
        # Initialize prompt optimization rules
        self._initialize_optimization_rules()
    
    def _initialize_model_registry(self):
        """Initialize the registry of available AI models"""
        
        self.model_registry = {
            'image_generation': {
                'stable_diffusion_xl': {
                    'name': 'Stable Diffusion XL',
                    'strengths': ['photorealistic', 'detailed', 'high_resolution'],
                    'weaknesses': ['slow', 'resource_intensive'],
                    'best_for': ['portraits', 'landscapes', 'detailed_art'],
                    'performance_score': 0.85,
                    'speed_score': 0.6,
                    'quality_score': 0.9,
                    'cost_score': 0.7
                },
                'midjourney_style': {
                    'name': 'Midjourney Style Model',
                    'strengths': ['artistic', 'creative', 'stylized'],
                    'weaknesses': ['less_photorealistic', 'inconsistent'],
                    'best_for': ['concept_art', 'illustrations', 'creative_designs'],
                    'performance_score': 0.8,
                    'speed_score': 0.8,
                    'quality_score': 0.85,
                    'cost_score': 0.8
                },
                'dalle_style': {
                    'name': 'DALL-E Style Model',
                    'strengths': ['versatile', 'text_understanding', 'creative'],
                    'weaknesses': ['moderate_quality', 'style_limitations'],
                    'best_for': ['general_purpose', 'text_integration', 'simple_concepts'],
                    'performance_score': 0.75,
                    'speed_score': 0.9,
                    'quality_score': 0.75,
                    'cost_score': 0.9
                }
            },
            'video_generation': {
                'mochi_1': {
                    'name': 'Mochi-1',
                    'strengths': ['high_quality', 'realistic_motion', 'detailed'],
                    'weaknesses': ['slow', 'resource_intensive', 'short_duration'],
                    'best_for': ['cinematic_shots', 'realistic_scenes', 'high_quality_content'],
                    'performance_score': 0.9,
                    'speed_score': 0.4,
                    'quality_score': 0.95,
                    'cost_score': 0.5
                },
                'cogvideox_5b': {
                    'name': 'CogVideoX-5B',
                    'strengths': ['fast', 'efficient', 'good_motion'],
                    'weaknesses': ['lower_quality', 'limited_styles'],
                    'best_for': ['quick_prototypes', 'simple_animations', 'fast_turnaround'],
                    'performance_score': 0.7,
                    'speed_score': 0.9,
                    'quality_score': 0.7,
                    'cost_score': 0.9
                },
                'animatediff': {
                    'name': 'AnimateDiff',
                    'strengths': ['animation', 'character_movement', 'stylized'],
                    'weaknesses': ['cartoon_style', 'limited_realism'],
                    'best_for': ['character_animation', 'cartoon_style', 'stylized_content'],
                    'performance_score': 0.75,
                    'speed_score': 0.8,
                    'quality_score': 0.8,
                    'cost_score': 0.8
                }
            },
            'audio_generation': {
                'bark_tts': {
                    'name': 'Bark TTS',
                    'strengths': ['natural_speech', 'emotional_expression', 'multilingual'],
                    'weaknesses': ['slow', 'resource_intensive'],
                    'best_for': ['natural_speech', 'voice_acting', 'emotional_content'],
                    'performance_score': 0.85,
                    'speed_score': 0.5,
                    'quality_score': 0.9,
                    'cost_score': 0.6
                },
                'gtts': {
                    'name': 'Google TTS',
                    'strengths': ['fast', 'reliable', 'multilingual'],
                    'weaknesses': ['robotic', 'limited_expression'],
                    'best_for': ['quick_speech', 'announcements', 'simple_narration'],
                    'performance_score': 0.7,
                    'speed_score': 0.95,
                    'quality_score': 0.6,
                    'cost_score': 0.95
                },
                'musicgen': {
                    'name': 'MusicGen',
                    'strengths': ['music_generation', 'style_variety', 'creative'],
                    'weaknesses': ['slow', 'limited_duration'],
                    'best_for': ['background_music', 'creative_audio', 'soundtracks'],
                    'performance_score': 0.8,
                    'speed_score': 0.6,
                    'quality_score': 0.85,
                    'cost_score': 0.7
                }
            },
            'game_creation': {
                'advanced_game_engine': {
                    'name': 'Advanced Game Engine',
                    'strengths': ['complex_games', 'rich_features', 'professional'],
                    'weaknesses': ['slow', 'complex', 'resource_intensive'],
                    'best_for': ['complex_games', 'professional_quality', 'rich_gameplay'],
                    'performance_score': 0.9,
                    'speed_score': 0.4,
                    'quality_score': 0.95,
                    'cost_score': 0.5
                },
                'simple_game_engine': {
                    'name': 'Simple Game Engine',
                    'strengths': ['fast', 'simple', 'reliable'],
                    'weaknesses': ['limited_features', 'basic_graphics'],
                    'best_for': ['simple_games', 'quick_prototypes', 'educational_games'],
                    'performance_score': 0.7,
                    'speed_score': 0.9,
                    'quality_score': 0.6,
                    'cost_score': 0.9
                }
            },
            'chat_assistance': {
                'groq_llama': {
                    'name': 'Groq Llama 3.1',
                    'strengths': ['fast', 'intelligent', 'versatile'],
                    'weaknesses': ['token_limits', 'context_limitations'],
                    'best_for': ['general_chat', 'quick_responses', 'versatile_tasks'],
                    'performance_score': 0.85,
                    'speed_score': 0.95,
                    'quality_score': 0.8,
                    'cost_score': 0.8
                }
            }
        }
    
    def _initialize_optimization_rules(self):
        """Initialize prompt optimization rules for different services"""
        
        self.optimization_rules = {
            'image_generation': {
                'quality_enhancers': [
                    'high quality', 'detailed', 'sharp', 'crisp', 'professional',
                    '4k', 'ultra detailed', 'masterpiece', 'best quality'
                ],
                'style_keywords': [
                    'photorealistic', 'artistic', 'digital art', 'oil painting',
                    'watercolor', 'sketch', 'cartoon', 'anime', 'realistic'
                ],
                'lighting_terms': [
                    'dramatic lighting', 'soft lighting', 'natural lighting',
                    'studio lighting', 'golden hour', 'cinematic lighting'
                ],
                'composition_terms': [
                    'rule of thirds', 'centered composition', 'dynamic angle',
                    'close-up', 'wide shot', 'portrait', 'landscape'
                ],
                'negative_prompts': [
                    'blurry', 'low quality', 'pixelated', 'distorted',
                    'ugly', 'deformed', 'bad anatomy', 'worst quality'
                ]
            },
            'video_generation': {
                'motion_enhancers': [
                    'smooth motion', 'fluid movement', 'cinematic',
                    'dynamic camera', 'stable footage', 'professional cinematography'
                ],
                'quality_enhancers': [
                    'high definition', '4k video', 'crisp footage',
                    'professional quality', 'cinematic quality'
                ],
                'style_keywords': [
                    'cinematic style', 'documentary style', 'artistic',
                    'realistic', 'animated', 'time-lapse', 'slow motion'
                ],
                'scene_descriptors': [
                    'well-lit scene', 'beautiful scenery', 'detailed environment',
                    'atmospheric', 'immersive', 'engaging visuals'
                ]
            },
            'audio_generation': {
                'quality_enhancers': [
                    'clear audio', 'high quality', 'professional recording',
                    'studio quality', 'crisp sound', 'well-mixed'
                ],
                'voice_descriptors': [
                    'natural voice', 'expressive', 'clear pronunciation',
                    'engaging tone', 'professional narrator', 'warm voice'
                ],
                'music_styles': [
                    'upbeat', 'melodic', 'harmonic', 'rhythmic',
                    'atmospheric', 'energetic', 'calming', 'dramatic'
                ]
            },
            'game_creation': {
                'engagement_enhancers': [
                    'engaging gameplay', 'addictive mechanics', 'fun',
                    'challenging', 'rewarding', 'interactive'
                ],
                'quality_descriptors': [
                    'polished', 'professional', 'well-designed',
                    'intuitive controls', 'smooth gameplay', 'responsive'
                ],
                'game_types': [
                    'arcade style', 'puzzle game', 'action game',
                    'strategy game', 'casual game', 'retro style'
                ]
            }
        }
    
    def select_optimal_model(self, service_type, prompt, user_id=None, requirements=None):
        """Select the optimal AI model based on prompt and requirements"""
        try:
            if service_type not in self.model_registry:
                return {
                    'error': f'Service type {service_type} not supported',
                    'available_services': list(self.model_registry.keys())
                }
            
            available_models = self.model_registry[service_type]
            model_scores = {}
            
            # Analyze prompt to determine requirements
            prompt_analysis = self._analyze_prompt(prompt, service_type)
            
            # Score each model
            for model_id, model_info in available_models.items():
                score = self._calculate_model_score(
                    model_info, prompt_analysis, requirements, user_id
                )
                model_scores[model_id] = score
            
            # Select best model
            best_model_id = max(model_scores, key=model_scores.get)
            best_model = available_models[best_model_id]
            
            # Update usage stats
            self.model_usage_stats[f"{service_type}_{best_model_id}"] += 1
            
            return {
                'selected_model': {
                    'id': best_model_id,
                    'name': best_model['name'],
                    'info': best_model,
                    'selection_score': round(model_scores[best_model_id], 3)
                },
                'all_scores': {
                    model_id: round(score, 3) 
                    for model_id, score in model_scores.items()
                },
                'prompt_analysis': prompt_analysis,
                'selection_reasoning': self._get_selection_reasoning(
                    best_model_id, best_model, prompt_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error selecting optimal model: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_prompt(self, prompt, service_type):
        """Analyze prompt to understand requirements"""
        try:
            prompt_lower = prompt.lower()
            analysis = {
                'keywords': prompt_lower.split(),
                'length': len(prompt),
                'complexity': 'simple',
                'style_indicators': [],
                'quality_requirements': 'standard',
                'speed_priority': False,
                'specific_features': []
            }
            
            # Determine complexity
            if len(prompt) > 100:
                analysis['complexity'] = 'complex'
            elif len(prompt) > 50:
                analysis['complexity'] = 'medium'
            
            # Check for quality requirements
            quality_keywords = ['high quality', 'professional', 'detailed', '4k', 'hd', 'masterpiece']
            if any(keyword in prompt_lower for keyword in quality_keywords):
                analysis['quality_requirements'] = 'high'
            
            # Check for speed priority
            speed_keywords = ['quick', 'fast', 'simple', 'basic']
            if any(keyword in prompt_lower for keyword in speed_keywords):
                analysis['speed_priority'] = True
            
            # Service-specific analysis
            if service_type == 'image_generation':
                analysis['style_indicators'] = self._extract_image_style_indicators(prompt_lower)
            elif service_type == 'video_generation':
                analysis['style_indicators'] = self._extract_video_style_indicators(prompt_lower)
            elif service_type == 'audio_generation':
                analysis['style_indicators'] = self._extract_audio_style_indicators(prompt_lower)
            elif service_type == 'game_creation':
                analysis['style_indicators'] = self._extract_game_style_indicators(prompt_lower)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing prompt: {str(e)}")
            return {'error': str(e)}
    
    def _extract_image_style_indicators(self, prompt):
        """Extract style indicators for image generation"""
        style_indicators = []
        
        style_keywords = {
            'photorealistic': ['photorealistic', 'realistic', 'photo', 'real'],
            'artistic': ['artistic', 'art', 'painting', 'drawing'],
            'cartoon': ['cartoon', 'animated', 'anime', 'comic'],
            'abstract': ['abstract', 'surreal', 'conceptual'],
            'vintage': ['vintage', 'retro', 'old', 'classic']
        }
        
        for style, keywords in style_keywords.items():
            if any(keyword in prompt for keyword in keywords):
                style_indicators.append(style)
        
        return style_indicators
    
    def _extract_video_style_indicators(self, prompt):
        """Extract style indicators for video generation"""
        style_indicators = []
        
        style_keywords = {
            'cinematic': ['cinematic', 'movie', 'film', 'dramatic'],
            'realistic': ['realistic', 'real', 'documentary'],
            'animated': ['animated', 'cartoon', 'animation'],
            'time_lapse': ['time-lapse', 'timelapse', 'fast'],
            'slow_motion': ['slow motion', 'slow-mo', 'slowmo']
        }
        
        for style, keywords in style_keywords.items():
            if any(keyword in prompt for keyword in keywords):
                style_indicators.append(style)
        
        return style_indicators
    
    def _extract_audio_style_indicators(self, prompt):
        """Extract style indicators for audio generation"""
        style_indicators = []
        
        style_keywords = {
            'speech': ['speech', 'voice', 'narration', 'speaking'],
            'music': ['music', 'song', 'melody', 'beat'],
            'electronic': ['electronic', 'techno', 'edm', 'synth'],
            'classical': ['classical', 'orchestra', 'piano', 'violin'],
            'ambient': ['ambient', 'atmospheric', 'calm', 'relaxing']
        }
        
        for style, keywords in style_keywords.items():
            if any(keyword in prompt for keyword in keywords):
                style_indicators.append(style)
        
        return style_indicators
    
    def _extract_game_style_indicators(self, prompt):
        """Extract style indicators for game creation"""
        style_indicators = []
        
        style_keywords = {
            'arcade': ['arcade', 'retro', 'classic', 'simple'],
            'puzzle': ['puzzle', 'brain', 'logic', 'thinking'],
            'action': ['action', 'fast', 'shooting', 'fighting'],
            'strategy': ['strategy', 'planning', 'tactical', 'turn-based'],
            'casual': ['casual', 'easy', 'relaxing', 'simple']
        }
        
        for style, keywords in style_keywords.items():
            if any(keyword in prompt for keyword in keywords):
                style_indicators.append(style)
        
        return style_indicators
    
    def _calculate_model_score(self, model_info, prompt_analysis, requirements, user_id):
        """Calculate score for a model based on prompt and requirements"""
        try:
            score = 0.0
            
            # Base performance score (40% weight)
            score += model_info['performance_score'] * 0.4
            
            # Quality vs Speed trade-off (30% weight)
            if prompt_analysis.get('quality_requirements') == 'high':
                score += model_info['quality_score'] * 0.3
            elif prompt_analysis.get('speed_priority'):
                score += model_info['speed_score'] * 0.3
            else:
                # Balanced approach
                score += (model_info['quality_score'] + model_info['speed_score']) / 2 * 0.3
            
            # Style matching (20% weight)
            style_match_score = self._calculate_style_match(
                model_info, prompt_analysis.get('style_indicators', [])
            )
            score += style_match_score * 0.2
            
            # User preference (10% weight)
            if user_id:
                user_preference_score = self._get_user_preference_score(
                    user_id, model_info
                )
                score += user_preference_score * 0.1
            
            # Cost consideration (bonus/penalty)
            if requirements and requirements.get('cost_priority'):
                score += (model_info['cost_score'] - 0.5) * 0.1
            
            return max(0.0, min(1.0, score))  # Normalize to 0-1
            
        except Exception as e:
            logger.error(f"Error calculating model score: {str(e)}")
            return 0.5  # Default neutral score
    
    def _calculate_style_match(self, model_info, style_indicators):
        """Calculate how well a model matches the required styles"""
        if not style_indicators:
            return 0.5  # Neutral score if no style indicators
        
        model_strengths = model_info.get('strengths', [])
        model_best_for = model_info.get('best_for', [])
        
        match_score = 0.0
        total_indicators = len(style_indicators)
        
        for indicator in style_indicators:
            # Check if indicator matches model strengths or best_for
            if (indicator in model_strengths or 
                indicator in model_best_for or
                any(indicator in strength for strength in model_strengths) or
                any(indicator in best for best in model_best_for)):
                match_score += 1.0
        
        return match_score / total_indicators if total_indicators > 0 else 0.5
    
    def _get_user_preference_score(self, user_id, model_info):
        """Get user preference score for a model"""
        if user_id not in self.user_model_preferences:
            return 0.5  # Neutral score for new users
        
        user_prefs = self.user_model_preferences[user_id]
        model_name = model_info['name']
        
        # Return stored preference or neutral score
        return user_prefs.get(model_name, 0.5)
    
    def _get_selection_reasoning(self, model_id, model_info, prompt_analysis):
        """Generate reasoning for model selection"""
        reasons = []
        
        # Quality vs Speed reasoning
        if prompt_analysis.get('quality_requirements') == 'high':
            if model_info['quality_score'] > 0.8:
                reasons.append(f"High quality requirements matched by {model_info['name']}'s excellent quality score")
        elif prompt_analysis.get('speed_priority'):
            if model_info['speed_score'] > 0.8:
                reasons.append(f"Speed priority matched by {model_info['name']}'s fast processing")
        
        # Style matching reasoning
        style_indicators = prompt_analysis.get('style_indicators', [])
        if style_indicators:
            matching_styles = [
                style for style in style_indicators 
                if style in model_info.get('best_for', [])
            ]
            if matching_styles:
                reasons.append(f"Style requirements ({', '.join(matching_styles)}) align with model strengths")
        
        # Strengths reasoning
        if model_info.get('strengths'):
            reasons.append(f"Model strengths: {', '.join(model_info['strengths'])}")
        
        return reasons
    
    def optimize_prompt(self, original_prompt, service_type, selected_model=None):
        """Optimize prompt for better results"""
        try:
            if service_type not in self.optimization_rules:
                return {
                    'original_prompt': original_prompt,
                    'optimized_prompt': original_prompt,
                    'optimizations': [],
                    'confidence': 0.0
                }
            
            optimized_prompt = original_prompt
            optimizations = []
            rules = self.optimization_rules[service_type]
            
            # Add quality enhancers if not present
            quality_enhancers = rules.get('quality_enhancers', [])
            if not any(enhancer.lower() in original_prompt.lower() for enhancer in quality_enhancers):
                # Add a random quality enhancer
                enhancer = random.choice(quality_enhancers[:3])  # Use top 3 most effective
                optimized_prompt += f", {enhancer}"
                optimizations.append(f"Added quality enhancer: '{enhancer}'")
            
            # Add style keywords if appropriate
            style_keywords = rules.get('style_keywords', [])
            prompt_lower = original_prompt.lower()
            
            # Check if any style is already specified
            has_style = any(style.lower() in prompt_lower for style in style_keywords)
            if not has_style and service_type in ['image_generation', 'video_generation']:
                # Suggest a default style based on content
                if any(word in prompt_lower for word in ['person', 'face', 'portrait']):
                    optimized_prompt += ", photorealistic"
                    optimizations.append("Added 'photorealistic' style for human subjects")
                elif any(word in prompt_lower for word in ['art', 'creative', 'design']):
                    optimized_prompt += ", artistic"
                    optimizations.append("Added 'artistic' style for creative content")
            
            # Service-specific optimizations
            if service_type == 'image_generation':
                optimizations.extend(self._optimize_image_prompt(optimized_prompt, rules))
            elif service_type == 'video_generation':
                optimizations.extend(self._optimize_video_prompt(optimized_prompt, rules))
            elif service_type == 'audio_generation':
                optimizations.extend(self._optimize_audio_prompt(optimized_prompt, rules))
            elif service_type == 'game_creation':
                optimizations.extend(self._optimize_game_prompt(optimized_prompt, rules))
            
            # Model-specific optimizations
            if selected_model:
                model_optimizations = self._apply_model_specific_optimizations(
                    optimized_prompt, selected_model, service_type
                )
                optimizations.extend(model_optimizations)
            
            # Calculate confidence based on number of optimizations
            confidence = min(len(optimizations) / 5, 1.0)  # Max confidence at 5+ optimizations
            
            return {
                'original_prompt': original_prompt,
                'optimized_prompt': optimized_prompt,
                'optimizations': optimizations,
                'confidence': round(confidence, 2),
                'service_type': service_type
            }
            
        except Exception as e:
            logger.error(f"Error optimizing prompt: {str(e)}")
            return {
                'original_prompt': original_prompt,
                'optimized_prompt': original_prompt,
                'optimizations': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _optimize_image_prompt(self, prompt, rules):
        """Apply image-specific optimizations"""
        optimizations = []
        prompt_lower = prompt.lower()
        
        # Add lighting if not specified
        lighting_terms = rules.get('lighting_terms', [])
        if not any(term.lower() in prompt_lower for term in lighting_terms):
            lighting = random.choice(['soft lighting', 'natural lighting', 'dramatic lighting'])
            optimizations.append(f"Consider adding lighting: '{lighting}'")
        
        # Add composition guidance
        composition_terms = rules.get('composition_terms', [])
        if not any(term.lower() in prompt_lower for term in composition_terms):
            composition = random.choice(['rule of thirds', 'centered composition'])
            optimizations.append(f"Consider composition: '{composition}'")
        
        return optimizations
    
    def _optimize_video_prompt(self, prompt, rules):
        """Apply video-specific optimizations"""
        optimizations = []
        prompt_lower = prompt.lower()
        
        # Add motion descriptors
        motion_enhancers = rules.get('motion_enhancers', [])
        if not any(enhancer.lower() in prompt_lower for enhancer in motion_enhancers):
            motion = random.choice(['smooth motion', 'cinematic'])
            optimizations.append(f"Consider adding motion quality: '{motion}'")
        
        return optimizations
    
    def _optimize_audio_prompt(self, prompt, rules):
        """Apply audio-specific optimizations"""
        optimizations = []
        prompt_lower = prompt.lower()
        
        # Add voice quality descriptors for speech
        if any(word in prompt_lower for word in ['speech', 'voice', 'narration']):
            voice_descriptors = rules.get('voice_descriptors', [])
            if not any(desc.lower() in prompt_lower for desc in voice_descriptors):
                voice_quality = random.choice(['natural voice', 'clear pronunciation'])
                optimizations.append(f"Consider voice quality: '{voice_quality}'")
        
        return optimizations
    
    def _optimize_game_prompt(self, prompt, rules):
        """Apply game-specific optimizations"""
        optimizations = []
        prompt_lower = prompt.lower()
        
        # Add engagement enhancers
        engagement_enhancers = rules.get('engagement_enhancers', [])
        if not any(enhancer.lower() in prompt_lower for enhancer in engagement_enhancers):
            engagement = random.choice(['engaging gameplay', 'fun'])
            optimizations.append(f"Consider adding engagement: '{engagement}'")
        
        return optimizations
    
    def _apply_model_specific_optimizations(self, prompt, model_info, service_type):
        """Apply optimizations specific to the selected model"""
        optimizations = []
        
        # Get model strengths and optimize accordingly
        strengths = model_info.get('strengths', [])
        
        if 'photorealistic' in strengths and 'photorealistic' not in prompt.lower():
            optimizations.append("Model excels at photorealistic content - consider emphasizing realism")
        
        if 'artistic' in strengths and 'artistic' not in prompt.lower():
            optimizations.append("Model excels at artistic content - consider emphasizing creativity")
        
        if 'fast' in strengths:
            optimizations.append("Model is optimized for speed - suitable for quick iterations")
        
        return optimizations
    
    def record_model_performance(self, service_type, model_id, performance_data):
        """Record performance data for a model"""
        try:
            performance_record = {
                'timestamp': datetime.now().isoformat(),
                'service_type': service_type,
                'model_id': model_id,
                'rating': performance_data.get('rating', 0),
                'generation_time': performance_data.get('generation_time', 0),
                'user_satisfaction': performance_data.get('user_satisfaction', 0),
                'quality_score': performance_data.get('quality_score', 0)
            }
            
            self.model_performance[f"{service_type}_{model_id}"].append(performance_record)
            
            # Update model registry with new performance data
            if service_type in self.model_registry and model_id in self.model_registry[service_type]:
                self._update_model_scores(service_type, model_id)
            
            return performance_record
            
        except Exception as e:
            logger.error(f"Error recording model performance: {str(e)}")
            raise
    
    def _update_model_scores(self, service_type, model_id):
        """Update model scores based on recent performance"""
        try:
            performance_key = f"{service_type}_{model_id}"
            recent_performance = self.model_performance[performance_key][-10:]  # Last 10 records
            
            if len(recent_performance) >= 3:  # Need at least 3 data points
                avg_rating = np.mean([p['rating'] for p in recent_performance])
                avg_quality = np.mean([p['quality_score'] for p in recent_performance])
                avg_satisfaction = np.mean([p['user_satisfaction'] for p in recent_performance])
                
                # Update model registry
                model_info = self.model_registry[service_type][model_id]
                
                # Weighted update (70% old score, 30% new data)
                model_info['performance_score'] = (
                    model_info['performance_score'] * 0.7 + 
                    (avg_rating / 5.0) * 0.3
                )
                model_info['quality_score'] = (
                    model_info['quality_score'] * 0.7 + 
                    avg_quality * 0.3
                )
                
                logger.info(f"Updated scores for {service_type}_{model_id}")
            
        except Exception as e:
            logger.error(f"Error updating model scores: {str(e)}")
    
    def get_model_analytics(self):
        """Get analytics about model usage and performance"""
        try:
            analytics = {
                'usage_stats': dict(self.model_usage_stats),
                'performance_trends': {},
                'model_rankings': {},
                'recommendations': []
            }
            
            # Calculate performance trends
            for service_type, models in self.model_registry.items():
                analytics['performance_trends'][service_type] = {}
                model_scores = []
                
                for model_id, model_info in models.items():
                    performance_key = f"{service_type}_{model_id}"
                    recent_performance = self.model_performance[performance_key][-5:]
                    
                    if recent_performance:
                        avg_rating = np.mean([p['rating'] for p in recent_performance])
                        analytics['performance_trends'][service_type][model_id] = {
                            'avg_rating': round(avg_rating, 2),
                            'usage_count': self.model_usage_stats.get(performance_key, 0),
                            'performance_score': round(model_info['performance_score'], 3)
                        }
                        model_scores.append((model_id, model_info['performance_score']))
                
                # Rank models by performance
                model_scores.sort(key=lambda x: x[1], reverse=True)
                analytics['model_rankings'][service_type] = [
                    {'model_id': model_id, 'score': round(score, 3)}
                    for model_id, score in model_scores
                ]
            
            # Generate recommendations
            analytics['recommendations'] = self._generate_model_recommendations()
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting model analytics: {str(e)}")
            return {'error': str(e)}
    
    def _generate_model_recommendations(self):
        """Generate recommendations for model improvements"""
        recommendations = []
        
        # Analyze usage patterns
        total_usage = sum(self.model_usage_stats.values())
        if total_usage > 0:
            # Find underused high-performance models
            for service_type, models in self.model_registry.items():
                for model_id, model_info in models.items():
                    usage_key = f"{service_type}_{model_id}"
                    usage_count = self.model_usage_stats.get(usage_key, 0)
                    usage_percentage = usage_count / total_usage
                    
                    if model_info['performance_score'] > 0.8 and usage_percentage < 0.1:
                        recommendations.append(
                            f"Consider promoting {model_info['name']} - high performance but low usage"
                        )
        
        # Analyze performance trends
        for performance_key, performance_data in self.model_performance.items():
            if len(performance_data) >= 5:
                recent_ratings = [p['rating'] for p in performance_data[-5:]]
                avg_recent = np.mean(recent_ratings)
                
                if avg_recent < 3.0:
                    service_type, model_id = performance_key.split('_', 1)
                    model_name = self.model_registry[service_type][model_id]['name']
                    recommendations.append(
                        f"Review {model_name} - recent performance below average ({avg_recent:.1f}/5.0)"
                    )
        
        return recommendations[:5]  # Return top 5 recommendations

# Initialize adaptive model system
adaptive_system = AdaptiveModelSystem()

@adaptive_models_bp.route('/select-model', methods=['POST'])
def select_optimal_model():
    """Select optimal model for a request"""
    try:
        data = request.get_json()
        
        if not data or 'service_type' not in data or 'prompt' not in data:
            return jsonify({
                'success': False,
                'error': 'Service type and prompt are required'
            }), 400
        
        service_type = data['service_type']
        prompt = data['prompt']
        user_id = data.get('user_id')
        requirements = data.get('requirements', {})
        
        selection_result = adaptive_system.select_optimal_model(
            service_type, prompt, user_id, requirements
        )
        
        return jsonify({
            'success': True,
            'message': 'Model selection completed',
            'selection': selection_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error selecting model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@adaptive_models_bp.route('/optimize-prompt', methods=['POST'])
def optimize_prompt():
    """Optimize prompt for better results"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data or 'service_type' not in data:
            return jsonify({
                'success': False,
                'error': 'Prompt and service type are required'
            }), 400
        
        prompt = data['prompt']
        service_type = data['service_type']
        selected_model = data.get('selected_model')
        
        optimization_result = adaptive_system.optimize_prompt(
            prompt, service_type, selected_model
        )
        
        return jsonify({
            'success': True,
            'message': 'Prompt optimization completed',
            'optimization': optimization_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error optimizing prompt: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@adaptive_models_bp.route('/record-performance', methods=['POST'])
def record_model_performance():
    """Record model performance data"""
    try:
        data = request.get_json()
        
        if not data or 'service_type' not in data or 'model_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Service type and model ID are required'
            }), 400
        
        service_type = data['service_type']
        model_id = data['model_id']
        performance_data = data.get('performance_data', {})
        
        performance_record = adaptive_system.record_model_performance(
            service_type, model_id, performance_data
        )
        
        return jsonify({
            'success': True,
            'message': 'Performance data recorded',
            'record': performance_record,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error recording performance: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@adaptive_models_bp.route('/get-analytics', methods=['GET'])
def get_model_analytics():
    """Get model usage and performance analytics"""
    try:
        analytics = adaptive_system.get_model_analytics()
        
        return jsonify({
            'success': True,
            'message': 'Analytics retrieved successfully',
            'analytics': analytics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@adaptive_models_bp.route('/get-models/<service_type>', methods=['GET'])
def get_available_models(service_type):
    """Get available models for a service type"""
    try:
        if service_type not in adaptive_system.model_registry:
            return jsonify({
                'success': False,
                'error': f'Service type {service_type} not supported',
                'available_services': list(adaptive_system.model_registry.keys())
            }), 400
        
        models = adaptive_system.model_registry[service_type]
        
        return jsonify({
            'success': True,
            'message': f'Available models for {service_type}',
            'models': models,
            'count': len(models),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@adaptive_models_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'Adaptive Model System is operational',
        'status': 'healthy',
        'services_supported': list(adaptive_system.model_registry.keys()),
        'total_models': sum(len(models) for models in adaptive_system.model_registry.values()),
        'timestamp': datetime.now().isoformat()
    })

