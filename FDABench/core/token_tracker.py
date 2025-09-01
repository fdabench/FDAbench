"""
Token tracking system for FDABench.

This module provides comprehensive token usage tracking using actual costs
from API responses rather than estimating costs.
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class TokenTrackingEntry:
    """A single token tracking entry for LLM calls"""
    category: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: float
    model: str
    cost: Optional[float] = None  # Actual cost from API response
    phase: Optional[str] = None  # Phase: decision, execute, retry, generate


@dataclass
class PhaseTrackingEntry:
    """A phase-specific tracking entry"""
    phase: str  # decision, execute, retry, generate
    category: str
    start_time: float
    end_time: float
    duration: float
    operation_type: str  # llm_call, tool_execution, other


class TokenTracker:
    """Tracks token usage for all LLM calls with actual costs from API responses"""
    
    def __init__(self):
        self.entries: List[TokenTrackingEntry] = []
        self.session_start_time = time.time()
        # Phase tracking additions
        self.phase_entries: List[PhaseTrackingEntry] = []
        self.phase_timings: Dict[str, List[float]] = {
            'decision': [], 'execute': [], 'retry': [], 'generate': []
        }
        self.current_phase: Optional[str] = None
        self.phase_start_time: Optional[float] = None
    
    def track_call(self, category: str, input_tokens: int, output_tokens: int, 
                   model: str = "moonshotai/kimi-k2", cost: Optional[float] = None,
                   phase: Optional[str] = None) -> TokenTrackingEntry:
        """
        Track a single LLM call with actual cost from API response.
        
        Args:
            category: Category of the call (e.g., "sql_generate", "planning")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens  
            model: Model name used for the call
            cost: Actual cost from API response
            
        Returns:
            TokenTrackingEntry with tracked information
        """
        total_tokens = input_tokens + output_tokens
        
        entry = TokenTrackingEntry(
            category=category,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            timestamp=time.time(),
            model=model,
            cost=cost,
            phase=phase or self.current_phase
        )
        
        self.entries.append(entry)
        return entry
    
    def get_total_tokens(self) -> Dict[str, int]:
        """Get total input, output and combined tokens"""
        return {
            "input": sum(entry.input_tokens for entry in self.entries),
            "output": sum(entry.output_tokens for entry in self.entries),
            "total": sum(entry.total_tokens for entry in self.entries)
        }
    
    def get_total_cost(self) -> float:
        """Get total cost from API responses"""
        return sum(entry.cost for entry in self.entries if entry.cost is not None)
    
    def get_breakdown_by_category(self) -> Dict[str, Dict[str, Any]]:
        """Get token and cost breakdown by category"""
        breakdown = {}
        
        for entry in self.entries:
            if entry.category not in breakdown:
                breakdown[entry.category] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0
                }
            
            breakdown[entry.category]["calls"] += 1
            breakdown[entry.category]["input_tokens"] += entry.input_tokens
            breakdown[entry.category]["output_tokens"] += entry.output_tokens
            breakdown[entry.category]["total_tokens"] += entry.total_tokens
            
            if entry.cost is not None:
                breakdown[entry.category]["total_cost"] += entry.cost
        
        return breakdown
    
    def get_breakdown_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Get token and cost breakdown by model"""
        breakdown = {}
        
        for entry in self.entries:
            if entry.model not in breakdown:
                breakdown[entry.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0
                }
            
            breakdown[entry.model]["calls"] += 1
            breakdown[entry.model]["input_tokens"] += entry.input_tokens
            breakdown[entry.model]["output_tokens"] += entry.output_tokens
            breakdown[entry.model]["total_tokens"] += entry.total_tokens
            
            if entry.cost is not None:
                breakdown[entry.model]["total_cost"] += entry.cost
        
        return breakdown
    
    def get_token_summary(self) -> Dict[str, Any]:
        """Get complete token usage summary"""
        total_tokens = self.get_total_tokens()
        total_cost = self.get_total_cost()
        breakdown_by_category = self.get_breakdown_by_category()
        breakdown_by_model = self.get_breakdown_by_model()
        
        session_duration = time.time() - self.session_start_time
        
        return {
            "total_tokens": total_tokens["total"],
            "total_calls": len(self.entries),
            "session_duration_seconds": round(session_duration, 2),
            "token_usage": total_tokens,
            "total_cost": total_cost,
            "breakdown_by_category": breakdown_by_category,
            "breakdown_by_model": breakdown_by_model
        }
    
    def start_phase(self, phase: str):
        """Start tracking a specific phase"""
        # End current phase if exists
        self.end_current_phase()
        
        self.current_phase = phase
        self.phase_start_time = time.time()
    
    def end_current_phase(self):
        """End current phase tracking"""
        if self.current_phase and self.phase_start_time:
            duration = time.time() - self.phase_start_time
            self.phase_timings[self.current_phase].append(duration)
            
            # Create phase entry
            phase_entry = PhaseTrackingEntry(
                phase=self.current_phase,
                category="phase_timing",
                start_time=self.phase_start_time,
                end_time=time.time(),
                duration=duration,
                operation_type="phase"
            )
            self.phase_entries.append(phase_entry)
    
    def track_phase_operation(self, phase: str, category: str, duration: float, operation_type: str = "other"):
        """Track a non-LLM operation in a specific phase"""
        end_time = time.time()
        start_time = end_time - duration
        
        self.phase_timings[phase].append(duration)
        
        phase_entry = PhaseTrackingEntry(
            phase=phase,
            category=category,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            operation_type=operation_type
        )
        self.phase_entries.append(phase_entry)
    
    def get_phase_summary(self) -> Dict[str, Any]:
        """Get comprehensive phase-based statistics"""
        # End current phase if active
        self.end_current_phase()
        
        phase_stats = {}
        
        for phase in ['decision', 'execute', 'retry', 'generate']:
            # Get token entries for this phase
            phase_token_entries = [e for e in self.entries if e.phase == phase]
            
            # Get timing entries for this phase
            phase_timing_entries = [e for e in self.phase_entries if e.phase == phase]
            
            # Calculate statistics
            total_latency = sum(self.phase_timings[phase])
            llm_calls = len(phase_token_entries)
            total_operations = len(phase_timing_entries)
            
            phase_stats[phase] = {
                # Token statistics
                'input_tokens': sum(e.input_tokens for e in phase_token_entries),
                'output_tokens': sum(e.output_tokens for e in phase_token_entries),
                'total_tokens': sum(e.total_tokens for e in phase_token_entries),
                'cost': sum(e.cost or 0 for e in phase_token_entries),
                'llm_calls': llm_calls,
                
                # Timing statistics
                'latency_seconds': round(total_latency, 3),
                'operation_count': total_operations,
                'avg_latency': round(total_latency / total_operations if total_operations > 0 else 0, 3),
                'max_latency': round(max(self.phase_timings[phase]) if self.phase_timings[phase] else 0, 3),
                'min_latency': round(min(self.phase_timings[phase]) if self.phase_timings[phase] else 0, 3)
            }
        
        return phase_stats
    
    def get_phase_database_columns(self) -> Dict[str, Any]:
        """Get phase statistics formatted for database storage"""
        phase_summary = self.get_phase_summary()
        
        columns = {}
        for phase in ['decision', 'execute', 'retry', 'generate']:
            stats = phase_summary[phase]
            columns.update({
                f'{phase}_latency_seconds': stats['latency_seconds'],
                f'{phase}_input_tokens': stats['input_tokens'],
                f'{phase}_output_tokens': stats['output_tokens'],
                f'{phase}_total_tokens': stats['total_tokens'],
                f'{phase}_llm_calls': stats['llm_calls'],
                f'{phase}_operation_count': stats['operation_count'],
                f'{phase}_cost': round(stats['cost'], 6),
                f'{phase}_avg_latency': stats['avg_latency'],
                f'{phase}_max_latency': stats['max_latency']
            })
        
        return columns
    
    def reset(self):
        """Reset tracking data"""
        self.entries.clear()
        self.phase_entries.clear()
        for phase in self.phase_timings:
            self.phase_timings[phase].clear()
        self.current_phase = None
        self.phase_start_time = None
        self.session_start_time = time.time()
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export all tracking data to dictionary"""
        return {
            "entries": [
                {
                    "category": entry.category,
                    "input_tokens": entry.input_tokens,
                    "output_tokens": entry.output_tokens,
                    "total_tokens": entry.total_tokens,
                    "timestamp": entry.timestamp,
                    "model": entry.model,
                    "cost": entry.cost
                }
                for entry in self.entries
            ],
            "summary": self.get_token_summary()
        }