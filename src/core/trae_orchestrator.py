"""
Trae Orchestrator - Central coordination system for all trading agents
Manages agent communication, event routing, and state synchronization
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of events in the system"""
    SIGNAL_GENERATED = "signal_generated"
    RISK_ASSESSMENT = "risk_assessment"
    FLOW_ANALYSIS = "flow_analysis"
    POSITION_UPDATE = "position_update"
    ORDER_EXECUTED = "order_executed"
    TRADE_ORDER_REQUEST = "trade_order_request"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SWARM_CONSENSUS = "swarm_consensus"
    MARKET_DATA = "market_data"
    SYSTEM_STATUS = "system_status"
    CORRELATION_UPDATE = "correlation_update"
    LEARNING_UPDATE = "learning_update"

@dataclass
class Event:
    """Event structure for inter-agent communication"""
    type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=low, 5=high
    correlation_id: Optional[str] = None

@dataclass
class AgentStatus:
    """Status information for each agent"""
    name: str
    active: bool
    confidence: float
    last_update: datetime
    performance_score: float
    error_count: int = 0

class TraeOrchestrator:
    """Central orchestrator for all trading agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, Any] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        self.event_queue = asyncio.Queue()
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.state_file = Path(config.get("state_file", "data/state.json"))
        self.summary_file = Path(config.get("summary_file", "reports/summary.json"))
        
        # Performance tracking
        self.performance_metrics = {
            "total_signals": 0,
            "successful_trades": 0,
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0
        }
        
        # Event processing
        self.running = False
        self.event_processor_task = None
        
        # Ensure directories exist
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def start(self):
        """Start the orchestrator"""
        self.running = True
        self.event_processor_task = asyncio.create_task(self._process_events())
        await self._load_state()
        logger.info("Trae Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        self.running = False
        if self.event_processor_task:
            self.event_processor_task.cancel()
            try:
                await self.event_processor_task
            except asyncio.CancelledError:
                pass
        await self._save_state()
        logger.info("Trae Orchestrator stopped")
    
    def register_agent(self, name: str, agent: Any, confidence: float = 1.0):
        """Register an agent with the orchestrator"""
        self.agents[name] = agent
        self.agent_status[name] = AgentStatus(
            name=name,
            active=True,
            confidence=confidence,
            last_update=datetime.now(timezone.utc),
            performance_score=0.0
        )
        logger.info(f"Registered agent: {name}")
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event: Event):
        """Emit an event to the system"""
        await self.event_queue.put(event)
        logger.debug(f"Event emitted: {event.type.value} from {event.source}")
    
    async def _process_events(self):
        """Process events from the queue"""
        while self.running:
            try:
                # Get event with timeout to allow periodic cleanup
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Update agent status
                if event.source in self.agent_status:
                    self.agent_status[event.source].last_update = datetime.now(timezone.utc)
                
                # Route event to handlers
                if event.type in self.event_handlers:
                    for handler in self.event_handlers[event.type]:
                        try:
                            await handler(event)
                        except Exception as e:
                            logger.error(f"Error in event handler: {e}")
                            if event.source in self.agent_status:
                                self.agent_status[event.source].error_count += 1
                
                # Log important events
                if event.priority >= 3:
                    await self._log_event(event)
                
            except asyncio.TimeoutError:
                # Periodic maintenance
                await self._update_performance_metrics()
                await self._check_agent_health()
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _log_event(self, event: Event):
        """Log important events to state file"""
        try:
            log_entry = {
                "timestamp": event.timestamp.isoformat(),
                "type": event.type.value,
                "source": event.source,
                "data": event.data,
                "priority": event.priority
            }
            
            # Append to state file
            async with aiofiles.open(self.state_file, "a") as f:
                await f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    async def _update_performance_metrics(self):
        """Update system performance metrics"""
        try:
            # Calculate metrics from recent events and agent performance
            total_confidence = sum(status.confidence for status in self.agent_status.values())
            active_agents = sum(1 for status in self.agent_status.values() if status.active)
            
            if active_agents > 0:
                avg_confidence = total_confidence / active_agents
                self.performance_metrics["avg_confidence"] = avg_confidence
            
            # Update timestamp
            self.performance_metrics["last_update"] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    async def _check_agent_health(self):
        """Check health of all registered agents"""
        current_time = datetime.now(timezone.utc)
        
        for name, status in self.agent_status.items():
            # Check if agent is responsive
            time_since_update = (current_time - status.last_update).total_seconds()
            
            if time_since_update > 300:  # 5 minutes
                status.active = False
                logger.warning(f"Agent {name} appears inactive (last update: {time_since_update}s ago)")
            
            # Adjust confidence based on error rate
            if status.error_count > 10:
                status.confidence = max(0.1, status.confidence * 0.9)
                logger.warning(f"Reducing confidence for agent {name} due to errors")
    
    async def _save_state(self):
        """Save current state to file"""
        try:
            state = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agents": {
                    name: asdict(status) for name, status in self.agent_status.items()
                },
                "performance": self.performance_metrics,
                "config": self.config
            }
            
            async with aiofiles.open(self.state_file, "w") as f:
                await f.write(json.dumps(state, indent=2, default=str))
                
            logger.debug("State saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def _load_state(self):
        """Load state from file"""
        try:
            if self.state_file.exists():
                async with aiofiles.open(self.state_file, "r") as f:
                    content = await f.read()
                    state = json.loads(content)
                
                # Restore performance metrics
                if "performance" in state:
                    self.performance_metrics.update(state["performance"])
                
                logger.info("State loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    async def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Agent summary
            agent_summary = {}
            for name, status in self.agent_status.items():
                agent_summary[name] = {
                    "active": status.active,
                    "confidence": status.confidence,
                    "performance_score": status.performance_score,
                    "error_count": status.error_count,
                    "last_update": status.last_update.isoformat()
                }
            
            # System summary
            summary = {
                "timestamp": current_time.isoformat(),
                "system_status": "healthy" if any(s.active for s in self.agent_status.values()) else "degraded",
                "active_agents": sum(1 for s in self.agent_status.values() if s.active),
                "total_agents": len(self.agent_status),
                "avg_confidence": sum(s.confidence for s in self.agent_status.values()) / len(self.agent_status) if self.agent_status else 0,
                "performance_metrics": self.performance_metrics,
                "agents": agent_summary
            }
            
            # Save summary report
            async with aiofiles.open(self.summary_file, "w") as f:
                await f.write(json.dumps(summary, indent=2, default=str))
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return {}
    
    def get_agent_status(self, name: str) -> Optional[AgentStatus]:
        """Get status of a specific agent"""
        return self.agent_status.get(name)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        active_agents = sum(1 for s in self.agent_status.values() if s.active)
        total_agents = len(self.agent_status)
        
        return {
            "status": "healthy" if active_agents > 0 else "degraded",
            "active_agents": active_agents,
            "total_agents": total_agents,
            "health_percentage": (active_agents / total_agents * 100) if total_agents > 0 else 0,
            "avg_confidence": sum(s.confidence for s in self.agent_status.values()) / total_agents if total_agents > 0 else 0
        }
    
    async def update_agent_performance(self, agent_name: str, performance_score: float):
        """Update performance score for an agent"""
        if agent_name in self.agent_status:
            self.agent_status[agent_name].performance_score = performance_score
            self.agent_status[agent_name].last_update = datetime.now(timezone.utc)
    
    async def adjust_agent_confidence(self, agent_name: str, adjustment: float):
        """Adjust confidence level for an agent"""
        if agent_name in self.agent_status:
            current_confidence = self.agent_status[agent_name].confidence
            new_confidence = max(0.0, min(1.0, current_confidence + adjustment))
            self.agent_status[agent_name].confidence = new_confidence
            logger.info(f"Adjusted confidence for {agent_name}: {current_confidence:.2f} -> {new_confidence:.2f}")

# Global orchestrator instance
orchestrator: Optional[TraeOrchestrator] = None

def get_orchestrator() -> TraeOrchestrator:
    """Get the global orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        raise RuntimeError("Orchestrator not initialized. Call initialize_orchestrator() first.")
    return orchestrator

def initialize_orchestrator(config: Dict[str, Any]) -> TraeOrchestrator:
    """Initialize the global orchestrator"""
    global orchestrator
    orchestrator = TraeOrchestrator(config)
    return orchestrator