from flask import Blueprint, request, jsonify
import requests
from bs4 import BeautifulSoup
import json
import time
import random
from urllib.parse import urljoin, urlparse
import logging
from datetime import datetime, timedelta

internet_crawler_bp = Blueprint('internet_crawler', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InternetLearningCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.learning_data = {
            'art_styles': [],
            'game_trends': [],
            'video_techniques': [],
            'audio_patterns': [],
            'user_preferences': []
        }
    
    def crawl_art_platforms(self):
        """Crawl art platforms for style learning"""
        art_sources = [
            {
                'name': 'Unsplash',
                'url': 'https://unsplash.com/s/photos/art',
                'type': 'photography'
            },
            {
                'name': 'DeviantArt',
                'url': 'https://www.deviantart.com/popular-24-hours/',
                'type': 'digital_art'
            }
        ]
        
        art_data = []
        for source in art_sources:
            try:
                logger.info(f"Crawling {source['name']} for art styles...")
                response = self.session.get(source['url'], timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract art metadata
                    if source['name'] == 'Unsplash':
                        images = soup.find_all('img', {'data-test': 'photo-grid-single-img'})
                        for img in images[:10]:  # Limit to 10 images
                            alt_text = img.get('alt', '')
                            if alt_text:
                                art_data.append({
                                    'source': source['name'],
                                    'type': source['type'],
                                    'description': alt_text,
                                    'timestamp': datetime.now().isoformat()
                                })
                    
                    # Add delay to be respectful
                    time.sleep(random.uniform(1, 3))
                    
            except Exception as e:
                logger.error(f"Error crawling {source['name']}: {str(e)}")
        
        self.learning_data['art_styles'].extend(art_data)
        return art_data
    
    def crawl_gaming_trends(self):
        """Crawl gaming platforms for trend analysis"""
        gaming_sources = [
            {
                'name': 'Steam',
                'url': 'https://store.steampowered.com/charts/topselling',
                'type': 'game_sales'
            }
        ]
        
        gaming_data = []
        for source in gaming_sources:
            try:
                logger.info(f"Crawling {source['name']} for gaming trends...")
                response = self.session.get(source['url'], timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract game titles and genres
                    game_elements = soup.find_all('a', class_='tab_row')
                    for game in game_elements[:10]:  # Limit to top 10
                        title_elem = game.find('div', class_='tab_item_name')
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            gaming_data.append({
                                'source': source['name'],
                                'type': source['type'],
                                'title': title,
                                'timestamp': datetime.now().isoformat()
                            })
                    
                    time.sleep(random.uniform(1, 3))
                    
            except Exception as e:
                logger.error(f"Error crawling {source['name']}: {str(e)}")
        
        self.learning_data['game_trends'].extend(gaming_data)
        return gaming_data
    
    def crawl_video_trends(self):
        """Crawl video platforms for technique analysis"""
        # Note: YouTube API would be better, but this demonstrates the concept
        video_data = []
        
        try:
            # Simulate video trend analysis
            trending_topics = [
                'cinematic video techniques',
                'popular video styles 2025',
                'viral video elements',
                'video composition trends'
            ]
            
            for topic in trending_topics:
                video_data.append({
                    'source': 'Video Analysis',
                    'type': 'trend_analysis',
                    'topic': topic,
                    'relevance_score': random.uniform(0.7, 1.0),
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error analyzing video trends: {str(e)}")
        
        self.learning_data['video_techniques'].extend(video_data)
        return video_data
    
    def crawl_audio_patterns(self):
        """Analyze audio and music trends"""
        audio_data = []
        
        try:
            # Simulate audio pattern analysis
            audio_trends = [
                {'genre': 'electronic', 'popularity': 0.85, 'characteristics': 'synthetic, rhythmic, energetic'},
                {'genre': 'ambient', 'popularity': 0.72, 'characteristics': 'atmospheric, calm, spacious'},
                {'genre': 'classical', 'popularity': 0.68, 'characteristics': 'orchestral, harmonic, structured'},
                {'genre': 'rock', 'popularity': 0.75, 'characteristics': 'guitar-driven, powerful, dynamic'}
            ]
            
            for trend in audio_trends:
                audio_data.append({
                    'source': 'Audio Analysis',
                    'type': 'music_trend',
                    'genre': trend['genre'],
                    'popularity': trend['popularity'],
                    'characteristics': trend['characteristics'],
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error analyzing audio patterns: {str(e)}")
        
        self.learning_data['audio_patterns'].extend(audio_data)
        return audio_data
    
    def analyze_user_preferences(self, user_data):
        """Analyze user preference patterns"""
        preference_data = []
        
        try:
            # Analyze user interaction patterns
            if user_data:
                for interaction in user_data:
                    preference_data.append({
                        'source': 'User Interaction',
                        'type': 'preference_analysis',
                        'category': interaction.get('category', 'unknown'),
                        'rating': interaction.get('rating', 0),
                        'engagement_time': interaction.get('engagement_time', 0),
                        'timestamp': datetime.now().isoformat()
                    })
                    
        except Exception as e:
            logger.error(f"Error analyzing user preferences: {str(e)}")
        
        self.learning_data['user_preferences'].extend(preference_data)
        return preference_data
    
    def get_learning_insights(self):
        """Generate insights from collected learning data"""
        insights = {
            'art_style_trends': [],
            'gaming_preferences': [],
            'video_techniques': [],
            'audio_preferences': [],
            'overall_trends': []
        }
        
        try:
            # Analyze art style trends
            if self.learning_data['art_styles']:
                art_descriptions = [item['description'] for item in self.learning_data['art_styles']]
                common_words = self._extract_common_keywords(art_descriptions)
                insights['art_style_trends'] = common_words[:5]
            
            # Analyze gaming trends
            if self.learning_data['game_trends']:
                game_titles = [item['title'] for item in self.learning_data['game_trends']]
                gaming_keywords = self._extract_common_keywords(game_titles)
                insights['gaming_preferences'] = gaming_keywords[:5]
            
            # Analyze video techniques
            if self.learning_data['video_techniques']:
                video_topics = [item['topic'] for item in self.learning_data['video_techniques']]
                video_keywords = self._extract_common_keywords(video_topics)
                insights['video_techniques'] = video_keywords[:5]
            
            # Analyze audio preferences
            if self.learning_data['audio_patterns']:
                popular_genres = sorted(
                    self.learning_data['audio_patterns'],
                    key=lambda x: x['popularity'],
                    reverse=True
                )
                insights['audio_preferences'] = [genre['genre'] for genre in popular_genres[:3]]
            
            # Generate overall trends
            insights['overall_trends'] = [
                'User preference for high-quality visual content',
                'Growing interest in interactive gaming experiences',
                'Demand for personalized AI-generated content',
                'Trend towards multi-modal AI applications'
            ]
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
        
        return insights
    
    def _extract_common_keywords(self, text_list):
        """Extract common keywords from text list"""
        word_count = {}
        
        for text in text_list:
            words = text.lower().split()
            for word in words:
                # Filter out common words
                if len(word) > 3 and word not in ['the', 'and', 'for', 'with', 'this', 'that']:
                    word_count[word] = word_count.get(word, 0) + 1
        
        # Return top keywords
        return sorted(word_count.items(), key=lambda x: x[1], reverse=True)

# Initialize crawler
crawler = InternetLearningCrawler()

@internet_crawler_bp.route('/crawl-art-styles', methods=['POST'])
def crawl_art_styles():
    """Endpoint to crawl art platforms for style learning"""
    try:
        art_data = crawler.crawl_art_platforms()
        
        return jsonify({
            'success': True,
            'message': f'Successfully crawled {len(art_data)} art style samples',
            'data': art_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in crawl_art_styles: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@internet_crawler_bp.route('/crawl-gaming-trends', methods=['POST'])
def crawl_gaming_trends():
    """Endpoint to crawl gaming platforms for trend analysis"""
    try:
        gaming_data = crawler.crawl_gaming_trends()
        
        return jsonify({
            'success': True,
            'message': f'Successfully crawled {len(gaming_data)} gaming trends',
            'data': gaming_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in crawl_gaming_trends: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@internet_crawler_bp.route('/crawl-video-trends', methods=['POST'])
def crawl_video_trends():
    """Endpoint to crawl video platforms for technique analysis"""
    try:
        video_data = crawler.crawl_video_trends()
        
        return jsonify({
            'success': True,
            'message': f'Successfully analyzed {len(video_data)} video trends',
            'data': video_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in crawl_video_trends: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@internet_crawler_bp.route('/crawl-audio-patterns', methods=['POST'])
def crawl_audio_patterns():
    """Endpoint to analyze audio and music trends"""
    try:
        audio_data = crawler.crawl_audio_patterns()
        
        return jsonify({
            'success': True,
            'message': f'Successfully analyzed {len(audio_data)} audio patterns',
            'data': audio_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in crawl_audio_patterns: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@internet_crawler_bp.route('/analyze-user-preferences', methods=['POST'])
def analyze_user_preferences():
    """Endpoint to analyze user preference patterns"""
    try:
        user_data = request.get_json()
        preference_data = crawler.analyze_user_preferences(user_data.get('interactions', []))
        
        return jsonify({
            'success': True,
            'message': f'Successfully analyzed {len(preference_data)} user preferences',
            'data': preference_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_user_preferences: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@internet_crawler_bp.route('/get-learning-insights', methods=['GET'])
def get_learning_insights():
    """Endpoint to get learning insights from collected data"""
    try:
        insights = crawler.get_learning_insights()
        
        return jsonify({
            'success': True,
            'message': 'Successfully generated learning insights',
            'insights': insights,
            'data_summary': {
                'art_styles_count': len(crawler.learning_data['art_styles']),
                'game_trends_count': len(crawler.learning_data['game_trends']),
                'video_techniques_count': len(crawler.learning_data['video_techniques']),
                'audio_patterns_count': len(crawler.learning_data['audio_patterns']),
                'user_preferences_count': len(crawler.learning_data['user_preferences'])
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in get_learning_insights: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@internet_crawler_bp.route('/crawl-all', methods=['POST'])
def crawl_all():
    """Endpoint to perform comprehensive internet crawling"""
    try:
        logger.info("Starting comprehensive internet crawling...")
        
        # Crawl all sources
        art_data = crawler.crawl_art_platforms()
        gaming_data = crawler.crawl_gaming_trends()
        video_data = crawler.crawl_video_trends()
        audio_data = crawler.crawl_audio_patterns()
        
        # Generate insights
        insights = crawler.get_learning_insights()
        
        total_data_points = len(art_data) + len(gaming_data) + len(video_data) + len(audio_data)
        
        return jsonify({
            'success': True,
            'message': f'Successfully completed comprehensive crawling with {total_data_points} data points',
            'results': {
                'art_styles': len(art_data),
                'gaming_trends': len(gaming_data),
                'video_techniques': len(video_data),
                'audio_patterns': len(audio_data)
            },
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in crawl_all: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@internet_crawler_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'Internet Learning Crawler is operational',
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })
