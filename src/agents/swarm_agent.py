"""
Swarm Agent for AI consensus and signal validation
Uses multiple AI perspectives to validate trading signals
Migrated for global agent hub with orchestrator integration
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import httpx
from dataclasses import dataclass
from enum import Enum

from ..core.trae_orchestrator import TraeOrchestrator, Event, EventType
from datetime import timezone

logger = logging.getLogger(__name__)


class ConsensusLevel(Enum):
    """Consensus levels"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class SwarmVote:
    """Individual swarm member vote"""
    agent_id: str
    signal: str
    confidence: float
    reasoning: str
    risk_assessment: str
    timestamp: datetime


@dataclass
class ConsensusResult:
    """Swarm consensus result"""
    symbol: str
    consensus: ConsensusLevel
    confidence: float
    votes: List[SwarmVote]
    reasoning: str
    risk_factors: List[str]
    timestamp: datetime


class SwarmAgent:
    """Swarm agent for AI consensus"""
    
    def __init__(self, orchestrator: TraeOrchestrator):
        self.orchestrator = orchestrator
        self.consensus_cache: Dict[str, ConsensusResult] = {}
        self.cache_duration = timedelta(minutes=15)
        
        # Swarm configuration
        self.swarm_size = 5  # Number of AI perspectives
        self.consensus_threshold = 0.6  # Minimum agreement for consensus
        self.min_confidence = 0.7  # Minimum confidence for strong signals
        
        # AI perspectives/roles
        self.ai_perspectives = [
            {
                "id": "technical_analyst",
                "role": "Technical Analysis Expert",
                "focus": "Chart patterns, indicators, and technical signals",
                "bias": "technical"
            },
            {
                "id": "fundamental_analyst", 
                "role": "Fundamental Analysis Expert",
                "focus": "Economic data, company fundamentals, and market conditions",
                "bias": "fundamental"
            },
            {
                "id": "risk_manager",
                "role": "Risk Management Specialist",
                "focus": "Risk assessment, position sizing, and capital preservation",
                "bias": "conservative"
            },
            {
                "id": "momentum_trader",
                "role": "Momentum Trading Expert",
                "focus": "Price momentum, volume analysis, and trend following",
                "bias": "aggressive"
            },
            {
                "id": "contrarian_analyst",
                "role": "Contrarian Analysis Expert",
                "focus": "Market sentiment extremes and reversal opportunities",
                "bias": "contrarian"
            }
        ]
    
    async def get_consensus(
        self,
        symbol: str,
        chart_analysis: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any],
        force_refresh: bool = False
    ) -> Optional[ConsensusResult]:
        """Get swarm consensus for a trading decision"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if not force_refresh and cache_key in self.consensus_cache:
                cached_result = self.consensus_cache[cache_key]
                if datetime.now() - cached_result.timestamp < self.cache_duration:
                    return cached_result
            
            # Gather votes from swarm
            votes = await self._gather_swarm_votes(symbol, chart_analysis, sentiment_data, market_data)
            
            if not votes:
                logger.warning(f"No votes received for {symbol}")
                return None
            
            # Calculate consensus
            consensus_result = await self._calculate_consensus(symbol, votes)
            
            # Cache result
            self.consensus_cache[cache_key] = consensus_result
            
            # Publish consensus via orchestrator
            await self.orchestrator.emit_event(Event(
                type=EventType.SWARM_CONSENSUS,
                source='SwarmAgent',
                data={
                    'symbol': symbol,
                    'consensus': consensus_result.consensus.value,
                    'confidence': consensus_result.confidence,
                    'reasoning': consensus_result.reasoning,
                    'risk_factors': consensus_result.risk_factors,
                    'vote_count': len(votes),
                    'timestamp': consensus_result.timestamp.isoformat()
                },
                timestamp=datetime.now(timezone.utc)
            ))
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"Failed to get consensus for {symbol}: {e}")
            return None
    
    async def _gather_swarm_votes(
        self,
        symbol: str,
        chart_analysis: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> List[SwarmVote]:
        """Gather votes from all swarm perspectives"""
        try:
            votes = []
            
            # Create tasks for parallel voting
            tasks = []
            for perspective in self.ai_perspectives:
                task = asyncio.create_task(
                    self._get_perspective_vote(perspective, symbol, chart_analysis, sentiment_data, market_data)
                )
                tasks.append(task)
            
            # Wait for all votes with timeout
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, SwarmVote):
                    votes.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Vote gathering failed: {result}")
            
            return votes
            
        except Exception as e:
            logger.error(f"Failed to gather swarm votes: {e}")
            return []
    
    async def _get_perspective_vote(
        self,
        perspective: Dict[str, Any],
        symbol: str,
        chart_analysis: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Optional[SwarmVote]:
        """Get vote from a specific AI perspective"""
        try:
            # Build prompt for this perspective
            prompt = await self._build_perspective_prompt(
                perspective, symbol, chart_analysis, sentiment_data, market_data
            )
            
            # Query AI (simplified - in real implementation would use actual AI service)
            response = await self._query_ai_mock(prompt, perspective)
            
            if not response:
                return None
            
            # Parse response
            vote_data = await self._parse_vote_response(response)
            if not vote_data:
                return None
            
            return SwarmVote(
                agent_id=perspective['id'],
                signal=vote_data['signal'],
                confidence=vote_data['confidence'],
                reasoning=vote_data['reasoning'],
                risk_assessment=vote_data['risk_assessment'],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get perspective vote from {perspective['id']}: {e}")
            return None
    
    async def _build_perspective_prompt(
        self,
        perspective: Dict[str, Any],
        symbol: str,
        chart_analysis: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> str:
        """Build AI prompt for specific perspective"""
        try:
            prompt = f"""
You are a {perspective['role']} with expertise in {perspective['focus']}.
Your trading bias is {perspective['bias']}.

Analyze the following data for {symbol} and provide your trading recommendation:

CHART ANALYSIS:
- Current Price: {chart_analysis.get('indicators', {}).get('current_price', 'N/A')}
- RSI: {chart_analysis.get('indicators', {}).get('rsi', 'N/A')}
- MACD: {chart_analysis.get('indicators', {}).get('macd', 'N/A')}
- Trend: {chart_analysis.get('trend', {}).get('direction', 'N/A')}
- Signals: {chart_analysis.get('signals', {}).get('overall_signal', 'N/A')}

SENTIMENT DATA:
- Overall Sentiment: {sentiment_data.get('overall_sentiment', 'N/A')}
- Confidence: {sentiment_data.get('confidence', 'N/A')}

MARKET DATA:
- Volume: {market_data.get('volume', 'N/A')}
- Volatility: {market_data.get('volatility', 'N/A')}

Based on your expertise and bias, provide:
1. SIGNAL: BUY/SELL/HOLD
2. CONFIDENCE: 0.0-1.0
3. REASONING: Brief explanation
4. RISK_ASSESSMENT: Key risks identified

Format your response as JSON:
{{
    "signal": "BUY/SELL/HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "Your reasoning here",
    "risk_assessment": "Key risks here"
}}
"""
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to build prompt: {e}")
            return ""
    
    async def _query_ai_mock(self, prompt: str, perspective: Dict[str, Any]) -> Optional[str]:
        """Mock AI query - in real implementation would use actual AI service"""
        try:
            # Simulate different AI responses based on perspective bias
            bias = perspective.get('bias', 'neutral')
            
            # Mock responses based on bias
            if bias == 'technical':
                return json.dumps({
                    "signal": "BUY",
                    "confidence": 0.75,
                    "reasoning": "Technical indicators show bullish momentum with RSI oversold and MACD turning positive",
                    "risk_assessment": "Risk of false breakout if volume doesn't confirm"
                })
            elif bias == 'fundamental':
                return json.dumps({
                    "signal": "HOLD",
                    "confidence": 0.65,
                    "reasoning": "Fundamentals are mixed with some positive indicators but economic uncertainty",
                    "risk_assessment": "Macro economic headwinds could impact performance"
                })
            elif bias == 'conservative':
                return json.dumps({
                    "signal": "HOLD",
                    "confidence": 0.80,
                    "reasoning": "Current market conditions suggest waiting for better risk-adjusted opportunities",
                    "risk_assessment": "High volatility increases position sizing risk"
                })
            elif bias == 'aggressive':
                return json.dumps({
                    "signal": "BUY",
                    "confidence": 0.85,
                    "reasoning": "Strong momentum signals suggest continuation of trend",
                    "risk_assessment": "Momentum could reverse quickly in volatile markets"
                })
            elif bias == 'contrarian':
                return json.dumps({
                    "signal": "SELL",
                    "confidence": 0.70,
                    "reasoning": "Market sentiment appears overly optimistic, suggesting potential reversal",
                    "risk_assessment": "Trend could continue longer than expected"
                })
            else:
                return json.dumps({
                    "signal": "HOLD",
                    "confidence": 0.60,
                    "reasoning": "Mixed signals suggest waiting for clearer direction",
                    "risk_assessment": "Uncertainty in multiple indicators"
                })
                
        except Exception as e:
            logger.error(f"Mock AI query failed: {e}")
            return None
    
    async def _parse_vote_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse AI response into structured vote data"""
        try:
            # Try to parse JSON response
            vote_data = json.loads(response)
            
            # Validate required fields
            required_fields = ['signal', 'confidence', 'reasoning', 'risk_assessment']
            if not all(field in vote_data for field in required_fields):
                logger.warning("Missing required fields in vote response")
                return None
            
            # Validate signal
            valid_signals = ['BUY', 'SELL', 'HOLD']
            if vote_data['signal'].upper() not in valid_signals:
                logger.warning(f"Invalid signal: {vote_data['signal']}")
                return None
            
            # Validate confidence
            confidence = float(vote_data['confidence'])
            if not 0.0 <= confidence <= 1.0:
                logger.warning(f"Invalid confidence: {confidence}")
                return None
            
            return {
                'signal': vote_data['signal'].upper(),
                'confidence': confidence,
                'reasoning': str(vote_data['reasoning']),
                'risk_assessment': str(vote_data['risk_assessment'])
            }
            
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response")
            return None
        except Exception as e:
            logger.error(f"Failed to parse vote response: {e}")
            return None
    
    async def _calculate_consensus(self, symbol: str, votes: List[SwarmVote]) -> ConsensusResult:
        """Calculate consensus from swarm votes"""
        try:
            if not votes:
                return ConsensusResult(
                    symbol=symbol,
                    consensus=ConsensusLevel.NEUTRAL,
                    confidence=0.0,
                    votes=[],
                    reasoning="No votes received",
                    risk_factors=[],
                    timestamp=datetime.now()
                )
            
            # Count votes by signal
            signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            weighted_signals = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
            total_confidence = 0.0
            
            for vote in votes:
                signal = vote.signal
                confidence = vote.confidence
                
                signal_counts[signal] += 1
                weighted_signals[signal] += confidence
                total_confidence += confidence
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(votes) if votes else 0.0
            
            # Determine consensus signal
            max_count = max(signal_counts.values())
            consensus_signals = [signal for signal, count in signal_counts.items() if count == max_count]
            
            # Handle ties
            if len(consensus_signals) > 1:
                # Use weighted approach for ties
                max_weight = max(weighted_signals.values())
                consensus_signal = [signal for signal, weight in weighted_signals.items() if weight == max_weight][0]
            else:
                consensus_signal = consensus_signals[0]
            
            # Calculate consensus strength
            agreement_ratio = max_count / len(votes)
            
            # Determine consensus level
            if consensus_signal == 'BUY':
                if agreement_ratio >= 0.8 and avg_confidence >= 0.8:
                    consensus_level = ConsensusLevel.STRONG_BUY
                elif agreement_ratio >= 0.6 and avg_confidence >= 0.7:
                    consensus_level = ConsensusLevel.BUY
                else:
                    consensus_level = ConsensusLevel.WEAK_BUY
            elif consensus_signal == 'SELL':
                if agreement_ratio >= 0.8 and avg_confidence >= 0.8:
                    consensus_level = ConsensusLevel.STRONG_SELL
                elif agreement_ratio >= 0.6 and avg_confidence >= 0.7:
                    consensus_level = ConsensusLevel.SELL
                else:
                    consensus_level = ConsensusLevel.WEAK_SELL
            else:
                consensus_level = ConsensusLevel.NEUTRAL
            
            # Generate reasoning
            reasoning = await self._generate_consensus_reasoning(votes, consensus_level)
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(votes)
            
            # Calculate final confidence
            final_confidence = avg_confidence * agreement_ratio
            
            return ConsensusResult(
                symbol=symbol,
                consensus=consensus_level,
                confidence=final_confidence,
                votes=votes,
                reasoning=reasoning,
                risk_factors=risk_factors,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate consensus: {e}")
            return ConsensusResult(
                symbol=symbol,
                consensus=ConsensusLevel.NEUTRAL,
                confidence=0.0,
                votes=votes,
                reasoning=f"Consensus calculation failed: {str(e)}",
                risk_factors=["Calculation error"],
                timestamp=datetime.now()
            )
    
    async def _generate_consensus_reasoning(self, votes: List[SwarmVote], consensus: ConsensusLevel) -> str:
        """Generate human-readable consensus reasoning"""
        try:
            if not votes:
                return "No votes available for analysis"
            
            # Count votes by signal
            signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            for vote in votes:
                signal_counts[vote.signal] += 1
            
            total_votes = len(votes)
            
            # Build reasoning
            reasoning_parts = []
            
            # Vote distribution
            for signal, count in signal_counts.items():
                if count > 0:
                    percentage = (count / total_votes) * 100
                    reasoning_parts.append(f"{count}/{total_votes} agents voted {signal} ({percentage:.0f}%)")
            
            # Key insights from votes
            key_insights = []
            for vote in votes:
                if vote.confidence >= 0.8:
                    key_insights.append(f"{vote.agent_id}: {vote.reasoning[:100]}...")
            
            reasoning = "Swarm Analysis: " + "; ".join(reasoning_parts)
            if key_insights:
                reasoning += ". Key insights: " + "; ".join(key_insights[:2])  # Limit to top 2
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            return "Reasoning generation failed"
    
    async def _identify_risk_factors(self, votes: List[SwarmVote]) -> List[str]:
        """Identify common risk factors from votes"""
        try:
            risk_factors = []
            risk_mentions = {}
            
            # Extract risk factors from votes
            for vote in votes:
                risk_text = vote.risk_assessment.lower()
                
                # Common risk keywords
                risk_keywords = [
                    'volatility', 'volume', 'liquidity', 'reversal', 'breakout',
                    'economic', 'macro', 'sentiment', 'momentum', 'trend'
                ]
                
                for keyword in risk_keywords:
                    if keyword in risk_text:
                        risk_mentions[keyword] = risk_mentions.get(keyword, 0) + 1
            
            # Select most mentioned risks
            sorted_risks = sorted(risk_mentions.items(), key=lambda x: x[1], reverse=True)
            
            for risk, count in sorted_risks[:5]:  # Top 5 risks
                if count >= 2:  # Mentioned by at least 2 agents
                    risk_factors.append(f"{risk.capitalize()} risk (mentioned by {count} agents)")
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Failed to identify risk factors: {e}")
            return ["Risk analysis failed"]
    
    async def validate_signal(
        self,
        symbol: str,
        proposed_signal: str,
        chart_analysis: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Validate a proposed trading signal against swarm consensus"""
        try:
            # Get swarm consensus
            consensus_result = await self.get_consensus(
                symbol, chart_analysis, sentiment_data, market_data
            )
            
            if not consensus_result:
                return False, 0.0, "Failed to get swarm consensus"
            
            # Check if proposed signal aligns with consensus
            consensus_signal = consensus_result.consensus.value
            
            # Map consensus levels to basic signals
            if consensus_signal in ['STRONG_BUY', 'BUY', 'WEAK_BUY']:
                swarm_signal = 'BUY'
            elif consensus_signal in ['STRONG_SELL', 'SELL', 'WEAK_SELL']:
                swarm_signal = 'SELL'
            else:
                swarm_signal = 'HOLD'
            
            # Validate alignment
            is_valid = proposed_signal.upper() == swarm_signal
            confidence = consensus_result.confidence
            
            validation_message = f"Swarm consensus: {consensus_signal} (confidence: {confidence:.2f}). "
            if is_valid:
                validation_message += f"Signal {proposed_signal} is VALIDATED by swarm."
            else:
                validation_message += f"Signal {proposed_signal} CONFLICTS with swarm consensus {swarm_signal}."
            
            return is_valid, confidence, validation_message
            
        except Exception as e:
            logger.error(f"Failed to validate signal: {e}")
            return False, 0.0, f"Signal validation failed: {str(e)}"
    
    async def get_swarm_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get summary of swarm consensus for multiple symbols"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'symbols_analyzed': len(symbols),
                'consensus_results': {},
                'overall_sentiment': 'NEUTRAL',
                'high_confidence_signals': []
            }
            
            # Collect consensus for each symbol (from cache if available)
            consensus_results = []
            for symbol in symbols:
                # Check cache for recent consensus
                for cache_key, cached_result in self.consensus_cache.items():
                    if symbol in cache_key and datetime.now() - cached_result.timestamp < self.cache_duration:
                        consensus_results.append(cached_result)
                        summary['consensus_results'][symbol] = {
                            'consensus': cached_result.consensus.value,
                            'confidence': cached_result.confidence,
                            'reasoning': cached_result.reasoning[:100] + "..." if len(cached_result.reasoning) > 100 else cached_result.reasoning
                        }
                        break
            
            # Identify high confidence signals
            for result in consensus_results:
                if result.confidence >= 0.8 and result.consensus != ConsensusLevel.NEUTRAL:
                    summary['high_confidence_signals'].append({
                        'symbol': result.symbol,
                        'consensus': result.consensus.value,
                        'confidence': result.confidence
                    })
            
            # Calculate overall sentiment
            if consensus_results:
                buy_signals = sum(1 for r in consensus_results if r.consensus.value in ['STRONG_BUY', 'BUY', 'WEAK_BUY'])
                sell_signals = sum(1 for r in consensus_results if r.consensus.value in ['STRONG_SELL', 'SELL', 'WEAK_SELL'])
                
                if buy_signals > sell_signals * 1.5:
                    summary['overall_sentiment'] = 'BULLISH'
                elif sell_signals > buy_signals * 1.5:
                    summary['overall_sentiment'] = 'BEARISH'
                else:
                    summary['overall_sentiment'] = 'NEUTRAL'
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get swarm summary: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'symbols_analyzed': 0,
                'consensus_results': {},
                'overall_sentiment': 'UNKNOWN',
                'high_confidence_signals': []
            }


# Global instance
_swarm_agent: Optional[SwarmAgent] = None


async def get_swarm_agent(orchestrator: TraeOrchestrator) -> SwarmAgent:
    """Get or create swarm agent instance"""
    global _swarm_agent
    if _swarm_agent is None:
        _swarm_agent = SwarmAgent(orchestrator)
    return _swarm_agent