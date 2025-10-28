"""
Sentiment Analysis Agent for market sentiment analysis
Migrated for global agent hub with orchestrator integration
"""
import asyncio
import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import httpx
import os

from ..core.trae_orchestrator import TraeOrchestrator, Event, EventType
from datetime import timezone

logger = logging.getLogger(__name__)


class SentimentAnalysisAgent:
    """Sentiment analysis agent using external APIs and news sources"""
    
    def __init__(self, orchestrator: TraeOrchestrator):
        self.orchestrator = orchestrator
        self.running = False
        self.sentiment_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=15)  # Cache sentiment for 15 minutes
        self.client: Optional[httpx.AsyncClient] = None
        
        # Configuration from environment
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        self.perplexity_base_url = os.getenv('PERPLEXITY_BASE_URL', 'https://api.perplexity.ai')
        self.perplexity_model = os.getenv('PERPLEXITY_MODEL', 'llama-3.1-sonar-small-128k-online')
        self.timeout = int(os.getenv('API_TIMEOUT', '30'))
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self.client is None:
            headers = {'Content-Type': 'application/json'}
            if self.perplexity_api_key:
                headers['Authorization'] = f'Bearer {self.perplexity_api_key}'
            
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=headers
            )
        return self.client
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def analyze_sentiment(self, symbol: str, context: str = "trading") -> Optional[Dict[str, Any]]:
        """Analyze sentiment for a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{context}"
            if self._is_cached(cache_key):
                return self.sentiment_cache[cache_key]
            
            # Get sentiment analysis
            sentiment_data = await self._get_sentiment_analysis(symbol, context)
            if not sentiment_data:
                # Fallback to mock analysis if API unavailable
                sentiment_data = await self._get_mock_sentiment(symbol, context)
            
            # Process and enhance the sentiment data
            processed_sentiment = await self._process_sentiment_data(sentiment_data, symbol)
            
            # Cache the result
            self.sentiment_cache[cache_key] = processed_sentiment
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
            
            # Publish analysis result via orchestrator
            await self.orchestrator.emit_event(Event(
                type=EventType.SENTIMENT_ANALYSIS,
                source='SentimentAnalysisAgent',
                data={
                    'symbol': symbol,
                    'sentiment_score': processed_sentiment.get('sentiment_score', 50),
                    'tradability_score': processed_sentiment.get('tradability_score', 50),
                    'confidence': processed_sentiment.get('confidence', 0.5),
                    'direction': processed_sentiment.get('direction', 'NEUTRAL'),
                    'strength': processed_sentiment.get('strength', 'WEAK'),
                    'recommendation': processed_sentiment.get('recommendation', 'HOLD'),
                    'risk_factors': processed_sentiment.get('risk_factors', []),
                    'sentiment_drivers': processed_sentiment.get('sentiment_drivers', []),
                    'timestamp': processed_sentiment.get('timestamp')
                },
                timestamp=datetime.now(timezone.utc)
            ))
            
            return processed_sentiment
            
        except Exception as e:
            logger.error(f"Failed to analyze sentiment for {symbol}: {e}")
            return None
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if sentiment is cached and not expired"""
        if cache_key not in self.sentiment_cache:
            return False
        
        if cache_key not in self.cache_expiry:
            return False
        
        return datetime.now() < self.cache_expiry[cache_key]
    
    async def _get_sentiment_analysis(self, symbol: str, context: str) -> Optional[Dict[str, Any]]:
        """Get sentiment analysis from external API"""
        try:
            if not self.perplexity_api_key:
                logger.warning("No Perplexity API key configured, using mock data")
                return None
            
            client = await self._get_client()
            
            # Construct the prompt for sentiment analysis
            prompt = self._build_sentiment_prompt(symbol, context)
            
            payload = {
                "model": self.perplexity_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial analyst specializing in market sentiment analysis. Provide objective, data-driven sentiment analysis based on recent market data, news, and financial indicators."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1,
                "top_p": 0.9,
                "return_citations": True,
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "day",
                "top_k": 0,
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 1
            }
            
            response = await client.post(
                f"{self.perplexity_base_url}/chat/completions",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_api_response(data)
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None
                
        except httpx.TimeoutException:
            logger.error(f"Timeout calling API for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error calling API for {symbol}: {e}")
            return None
    
    async def _get_mock_sentiment(self, symbol: str, context: str) -> Dict[str, Any]:
        """Generate mock sentiment data when API is unavailable"""
        try:
            # Generate realistic mock data based on symbol characteristics
            import random
            
            # Base sentiment varies by asset type
            if symbol.startswith(('EUR', 'GBP', 'USD', 'JPY')):
                # Forex pairs - more neutral
                base_sentiment = random.randint(45, 55)
                base_tradability = random.randint(70, 90)
            elif symbol.startswith(('BTC', 'ETH', 'ADA')):
                # Crypto - more volatile sentiment
                base_sentiment = random.randint(30, 70)
                base_tradability = random.randint(60, 85)
            else:
                # Stocks/indices - moderate sentiment
                base_sentiment = random.randint(40, 60)
                base_tradability = random.randint(65, 85)
            
            mock_drivers = [
                "Technical momentum indicators",
                "Market volatility patterns",
                "Volume analysis trends",
                "Economic calendar events",
                "Risk sentiment shifts"
            ]
            
            return {
                'sentiment_score': base_sentiment,
                'tradability_score': base_tradability,
                'confidence_level': random.randint(60, 80),
                'sentiment_drivers': random.sample(mock_drivers, 3),
                'raw_content': f"Mock sentiment analysis for {symbol}",
                'citations': [],
                'timestamp': datetime.now().isoformat(),
                'source': 'mock_data'
            }
            
        except Exception as e:
            logger.error(f"Failed to generate mock sentiment: {e}")
            return {
                'sentiment_score': 50,
                'tradability_score': 50,
                'confidence_level': 50,
                'sentiment_drivers': [],
                'raw_content': f"Default sentiment for {symbol}",
                'citations': [],
                'timestamp': datetime.now().isoformat(),
                'source': 'default'
            }
    
    def _build_sentiment_prompt(self, symbol: str, context: str) -> str:
        """Build prompt for sentiment analysis"""
        base_prompt = f"""
        Analyze the current market sentiment and tradability for {symbol} in the context of {context}.
        
        Please provide a comprehensive analysis including:
        
        1. **Overall Sentiment Score** (0-100, where 0 is extremely bearish, 50 is neutral, 100 is extremely bullish)
        
        2. **Key Sentiment Drivers** (list the top 3-5 factors influencing current sentiment)
        
        3. **Recent News Impact** (how recent news/events are affecting sentiment)
        
        4. **Market Momentum** (current price action and volume trends)
        
        5. **Institutional Activity** (any notable institutional buying/selling)
        
        6. **Technical Sentiment** (how technical indicators align with fundamental sentiment)
        
        7. **Risk Factors** (key risks that could change sentiment)
        
        8. **Tradability Score** (0-100, considering liquidity, volatility, and market conditions)
        
        9. **Time Horizon** (short-term vs long-term sentiment outlook)
        
        10. **Confidence Level** (0-100, how confident you are in this analysis)
        
        Please format your response as a structured analysis with clear sections and specific numerical scores where requested.
        Focus on recent data (last 24-48 hours) and current market conditions.
        """
        
        return base_prompt.strip()
    
    def _parse_api_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse API response"""
        try:
            if 'choices' not in response_data or not response_data['choices']:
                return {}
            
            content = response_data['choices'][0]['message']['content']
            citations = response_data.get('citations', [])
            
            # Extract structured data from the response
            sentiment_data = {
                'raw_content': content,
                'citations': citations,
                'timestamp': datetime.now().isoformat(),
                'source': 'api_response'
            }
            
            # Try to extract numerical scores using simple parsing
            sentiment_data.update(self._extract_scores_from_content(content))
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
            return {}
    
    def _extract_scores_from_content(self, content: str) -> Dict[str, Any]:
        """Extract numerical scores from content using simple text parsing"""
        try:
            scores = {}
            
            # Extract sentiment score
            sentiment_match = re.search(r'sentiment.*?score.*?(\d+)', content, re.IGNORECASE)
            if sentiment_match:
                scores['sentiment_score'] = int(sentiment_match.group(1))
            
            # Extract tradability score
            tradability_match = re.search(r'tradability.*?score.*?(\d+)', content, re.IGNORECASE)
            if tradability_match:
                scores['tradability_score'] = int(tradability_match.group(1))
            
            # Extract confidence level
            confidence_match = re.search(r'confidence.*?level.*?(\d+)', content, re.IGNORECASE)
            if confidence_match:
                scores['confidence_level'] = int(confidence_match.group(1))
            
            # Extract key sentiment drivers
            drivers = []
            drivers_section = re.search(r'key sentiment drivers.*?:(.*?)(?=\n\d+\.|\n[A-Z]|\Z)', content, re.IGNORECASE | re.DOTALL)
            if drivers_section:
                driver_text = drivers_section.group(1)
                # Simple extraction of bullet points or numbered items
                driver_matches = re.findall(r'[-•*]\s*(.+?)(?=\n|$)', driver_text)
                drivers = [driver.strip() for driver in driver_matches[:5]]  # Top 5
            
            scores['sentiment_drivers'] = drivers
            
            return scores
            
        except Exception as e:
            logger.error(f"Error extracting scores from content: {e}")
            return {}
    
    async def _process_sentiment_data(self, sentiment_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process and enhance sentiment data"""
        try:
            processed = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'source': sentiment_data.get('source', 'unknown'),
                'raw_data': sentiment_data
            }
            
            # Extract and normalize scores
            sentiment_score = sentiment_data.get('sentiment_score', 50)
            tradability_score = sentiment_data.get('tradability_score', 50)
            confidence_level = sentiment_data.get('confidence_level', 50)
            
            # Normalize scores to 0-100 range
            sentiment_score = max(0, min(100, sentiment_score))
            tradability_score = max(0, min(100, tradability_score))
            confidence_level = max(0, min(100, confidence_level))
            
            # Calculate derived metrics
            processed.update({
                'sentiment_score': sentiment_score,
                'tradability_score': tradability_score,
                'confidence': confidence_level / 100.0,  # Normalize to 0-1
                'direction': self._get_sentiment_direction(sentiment_score),
                'strength': self._get_sentiment_strength(sentiment_score),
                'tradability_rating': self._get_tradability_rating(tradability_score),
                'recommendation': self._get_sentiment_recommendation(sentiment_score, tradability_score, confidence_level),
                'sentiment_drivers': sentiment_data.get('sentiment_drivers', []),
                'risk_assessment': self._assess_sentiment_risk(sentiment_score, confidence_level)
            })
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing sentiment data: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'sentiment_score': 50,
                'tradability_score': 50,
                'confidence': 0.5,
                'direction': 'NEUTRAL',
                'strength': 'WEAK',
                'tradability_rating': 'MODERATE',
                'recommendation': 'HOLD',
                'sentiment_drivers': [],
                'risk_assessment': {'level': 'MODERATE', 'factors': ['Analysis error']}
            }
    
    def _get_sentiment_direction(self, score: int) -> str:
        """Get sentiment direction from score"""
        if score >= 60:
            return 'BULLISH'
        elif score <= 40:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _get_sentiment_strength(self, score: int) -> str:
        """Get sentiment strength from score"""
        if score >= 80 or score <= 20:
            return 'STRONG'
        elif score >= 65 or score <= 35:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def _get_tradability_rating(self, score: int) -> str:
        """Get tradability rating from score"""
        if score >= 80:
            return 'EXCELLENT'
        elif score >= 65:
            return 'GOOD'
        elif score >= 50:
            return 'MODERATE'
        elif score >= 35:
            return 'POOR'
        else:
            return 'VERY_POOR'
    
    def _get_sentiment_recommendation(self, sentiment: int, tradability: int, confidence: int) -> str:
        """Get trading recommendation based on sentiment metrics"""
        try:
            # Weight factors
            weighted_score = (sentiment * 0.5) + (tradability * 0.3) + (confidence * 0.2)
            
            if weighted_score >= 70:
                return 'STRONG_BUY'
            elif weighted_score >= 60:
                return 'BUY'
            elif weighted_score >= 55:
                return 'WEAK_BUY'
            elif weighted_score >= 45:
                return 'HOLD'
            elif weighted_score >= 40:
                return 'WEAK_SELL'
            elif weighted_score >= 30:
                return 'SELL'
            else:
                return 'STRONG_SELL'
                
        except Exception as e:
            logger.error(f"Error calculating recommendation: {e}")
            return 'HOLD'
    
    def _assess_sentiment_risk(self, sentiment: int, confidence: int) -> Dict[str, Any]:
        """Assess sentiment-based risks"""
        try:
            risk_factors = []
            
            # Extreme sentiment risks
            if sentiment >= 85:
                risk_factors.append("Extremely bullish sentiment may indicate overextension")
            elif sentiment <= 15:
                risk_factors.append("Extremely bearish sentiment may indicate oversold conditions")
            
            # Confidence-based risks
            if confidence <= 30:
                risk_factors.append("Low confidence in sentiment analysis")
            elif confidence >= 90:
                risk_factors.append("Overconfidence may lead to complacency")
            
            # Neutral sentiment risks
            if 45 <= sentiment <= 55:
                risk_factors.append("Neutral sentiment may indicate lack of clear direction")
            
            # Determine overall risk level
            if len(risk_factors) >= 3 or confidence <= 25:
                risk_level = 'HIGH'
            elif len(risk_factors) >= 2 or confidence <= 40:
                risk_level = 'MODERATE'
            elif len(risk_factors) >= 1:
                risk_level = 'LOW'
            else:
                risk_level = 'MINIMAL'
            
            return {
                'level': risk_level,
                'factors': risk_factors,
                'confidence_risk': confidence <= 40,
                'sentiment_extreme': sentiment >= 85 or sentiment <= 15
            }
            
        except Exception as e:
            logger.error(f"Error assessing sentiment risk: {e}")
            return {
                'level': 'MODERATE',
                'factors': ['Risk assessment error'],
                'confidence_risk': True,
                'sentiment_extreme': False
            }
    
    async def get_tradability_score(self, symbol: str) -> Optional[float]:
        """Get tradability score for a symbol"""
        try:
            sentiment_data = await self.analyze_sentiment(symbol)
            if sentiment_data:
                return sentiment_data.get('tradability_score', 50) / 100.0
            return None
        except Exception as e:
            logger.error(f"Failed to get tradability score for {symbol}: {e}")
            return None
    
    async def get_market_sentiment_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market sentiment summary for multiple symbols"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'symbols_analyzed': len(symbols),
                'overall_sentiment': 'NEUTRAL',
                'average_sentiment_score': 50.0,
                'average_tradability_score': 50.0,
                'high_confidence_signals': [],
                'sentiment_distribution': {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0}
            }
            
            sentiment_scores = []
            tradability_scores = []
            
            # Analyze each symbol
            for symbol in symbols:
                sentiment_data = await self.analyze_sentiment(symbol)
                if sentiment_data:
                    sentiment_score = sentiment_data.get('sentiment_score', 50)
                    tradability_score = sentiment_data.get('tradability_score', 50)
                    confidence = sentiment_data.get('confidence', 0.5)
                    direction = sentiment_data.get('direction', 'NEUTRAL')
                    
                    sentiment_scores.append(sentiment_score)
                    tradability_scores.append(tradability_score)
                    
                    # Count sentiment distribution
                    summary['sentiment_distribution'][direction] += 1
                    
                    # Collect high confidence signals
                    if confidence >= 0.8:
                        summary['high_confidence_signals'].append({
                            'symbol': symbol,
                            'sentiment_score': sentiment_score,
                            'direction': direction,
                            'confidence': confidence,
                            'recommendation': sentiment_data.get('recommendation', 'HOLD')
                        })
            
            # Calculate averages
            if sentiment_scores:
                summary['average_sentiment_score'] = sum(sentiment_scores) / len(sentiment_scores)
                summary['average_tradability_score'] = sum(tradability_scores) / len(tradability_scores)
                
                # Determine overall sentiment
                avg_sentiment = summary['average_sentiment_score']
                if avg_sentiment >= 60:
                    summary['overall_sentiment'] = 'BULLISH'
                elif avg_sentiment <= 40:
                    summary['overall_sentiment'] = 'BEARISH'
                else:
                    summary['overall_sentiment'] = 'NEUTRAL'
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get market sentiment summary: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'symbols_analyzed': 0,
                'overall_sentiment': 'UNKNOWN',
                'average_sentiment_score': 50.0,
                'average_tradability_score': 50.0,
                'high_confidence_signals': [],
                'sentiment_distribution': {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0}
            }
    
    async def start_continuous_sentiment_analysis(self, symbols: List[str], interval: int = 900) -> None:
        """Start continuous sentiment analysis for symbols"""
        try:
            if self.running:
                logger.warning("Continuous sentiment analysis already running")
                return
            
            self.running = True
            logger.info(f"Starting continuous sentiment analysis for {len(symbols)} symbols")
            
            # Create tasks for each symbol
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(self._sentiment_worker(symbol, interval))
                tasks.append(task)
            
            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in continuous sentiment analysis: {e}")
        finally:
            self.running = False
    
    async def _sentiment_worker(self, symbol: str, interval: int) -> None:
        """Worker for continuous sentiment analysis"""
        try:
            while self.running:
                await self.analyze_sentiment(symbol)
                await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"Sentiment worker error for {symbol}: {e}")
    
    async def stop_continuous_analysis(self) -> None:
        """Stop continuous sentiment analysis"""
        try:
            self.running = False
            logger.info("Stopping continuous sentiment analysis")
        except Exception as e:
            logger.error(f"Error stopping continuous analysis: {e}")


# Global instance
_sentiment_agent: Optional[SentimentAnalysisAgent] = None


async def get_sentiment_agent(orchestrator: TraeOrchestrator) -> SentimentAnalysisAgent:
    """Get or create sentiment agent instance"""
    global _sentiment_agent
    if _sentiment_agent is None:
        _sentiment_agent = SentimentAnalysisAgent(orchestrator)
    return _sentiment_agent


async def close_sentiment_agent():
    """Close sentiment agent"""
    global _sentiment_agent
    if _sentiment_agent:
        await _sentiment_agent.close()
        _sentiment_agent = None