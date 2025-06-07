"""
MkDocs Macros for LLM Study Guide
Provides dynamic content generation and interactive components
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any


def define_env(env):
    """Define macros and variables for the MkDocs environment"""
    
    @env.macro
    def progress_bar(percentage: int, label: str = "", color: str = "primary") -> str:
        """Generate an interactive progress bar"""
        return f"""
        <div class="interactive-progress" data-percentage="{percentage}">
            <div class="progress-header">
                <span class="progress-label">{label}</span>
                <span class="progress-percentage">{percentage}%</span>
            </div>
            <div class="progress-bar-interactive">
                <div class="progress-fill-interactive" style="width: {percentage}%"></div>
            </div>
        </div>
        """
    
    @env.macro
    def study_checkpoint(title: str, checkpoint_id: str, description: str = "") -> str:
        """Generate a study checkpoint component"""
        return f"""
        <div class="study-checkpoint" data-checkpoint="{checkpoint_id}">
            <div class="checkpoint-content">
                <h4>{title}</h4>
                {f'<p>{description}</p>' if description else ''}
                <div class="checkpoint-actions">
                    <button class="checkpoint-btn" onclick="studyComponents.toggleCheckpoint('{checkpoint_id}', this)">
                        ⏳ Mark Complete
                    </button>
                    <button class="checkpoint-btn" onclick="studyComponents.addCheckpointNote('{checkpoint_id}')">
                        📝 Add Note
                    </button>
                </div>
            </div>
        </div>
        """
    
    @env.macro
    def week_card(week_num: int, title: str, duration: str, topics: List[str], status: str = "pending") -> str:
        """Generate a week overview card"""
        topics_html = "".join([f"<li>{topic}</li>" for topic in topics])
        status_icon = {
            "completed": "✅",
            "in-progress": "🟡", 
            "pending": "⏳"
        }.get(status, "⏳")
        
        return f"""
        <div class="week-card" data-week="{week_num}">
            <div class="week-header">
                <h3>Week {week_num}: {title}</h3>
                <span class="week-status">{status_icon} {status.title()}</span>
            </div>
            <div class="week-meta">
                <span class="week-duration">⏱️ {duration}</span>
                <span class="week-number">Week {week_num}</span>
            </div>
            <div class="week-topics">
                <h4>Key Topics:</h4>
                <ul>{topics_html}</ul>
            </div>
            <div class="week-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
            <div class="week-actions">
                <a href="week-{week_num}/" class="week-link">Start Week →</a>
            </div>
        </div>
        """
    
    @env.macro
    def healthcare_callout(title: str, content: str, type: str = "info") -> str:
        """Generate a healthcare-specific callout box"""
        icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "safety": "🛡️",
            "compliance": "📋",
            "clinical": "🩺"
        }
        
        return f"""
        <div class="healthcare-callout healthcare-{type}">
            <div class="callout-header">
                <span class="callout-icon">{icons.get(type, 'ℹ️')}</span>
                <h4>{title}</h4>
            </div>
            <div class="callout-content">
                {content}
            </div>
        </div>
        """
    
    @env.macro
    def apple_silicon_tip(content: str) -> str:
        """Generate an Apple Silicon optimization tip"""
        return f"""
        <div class="apple-silicon-tip">
            <div class="tip-header">
                <span class="tip-icon">🍎</span>
                <h4>Apple Silicon M3 Ultra Optimization</h4>
            </div>
            <div class="tip-content">
                {content}
            </div>
        </div>
        """
    
    @env.macro
    def math_derivation(title: str, steps: List[str]) -> str:
        """Generate an interactive mathematical derivation"""
        steps_html = ""
        for i, step in enumerate(steps):
            steps_html += f"""
            <div class="derivation-step" data-step="{i+1}">
                <div class="step-number">{i+1}</div>
                <div class="step-content">{step}</div>
            </div>
            """
        
        return f"""
        <div class="math-derivation">
            <div class="derivation-header">
                <h4>🧮 {title}</h4>
                <button class="derivation-toggle" onclick="toggleDerivation(this)">Show Steps</button>
            </div>
            <div class="derivation-steps" style="display: none;">
                {steps_html}
            </div>
        </div>
        """
    
    @env.macro
    def code_example(language: str, code: str, title: str = "", healthcare: bool = False) -> str:
        """Generate an enhanced code example with healthcare focus"""
        healthcare_class = "healthcare-code" if healthcare else ""
        
        return f"""
        <div class="enhanced-code-example {healthcare_class}">
            {f'<div class="code-title">{title}</div>' if title else ''}
            <div class="code-header">
                <span class="code-language">{language.upper()}</span>
                <div class="code-actions">
                    <button onclick="copyCode(this)" title="Copy code">📋</button>
                    <button onclick="runCode(this)" title="Run code">▶️</button>
                    {f'<span class="healthcare-badge">🏥 Healthcare</span>' if healthcare else ''}
                </div>
            </div>
            <pre><code class="language-{language}">{code}</code></pre>
        </div>
        """
    
    @env.macro
    def study_session_card(title: str, duration: str, objectives: List[str]) -> str:
        """Generate a study session planning card"""
        objectives_html = "".join([f"<li>{obj}</li>" for obj in objectives])
        
        return f"""
        <div class="study-session-card">
            <div class="session-header">
                <h3 class="session-title">{title}</h3>
                <span class="session-duration">{duration}</span>
            </div>
            <div class="session-objectives">
                <h4>Learning Objectives:</h4>
                <ul>{objectives_html}</ul>
            </div>
            <div class="session-actions">
                <button class="session-btn" onclick="startStudySession('{title}')">
                    ⏱️ Start Session
                </button>
                <button class="session-btn" onclick="planSession('{title}')">
                    📋 Plan Session
                </button>
            </div>
        </div>
        """
    
    @env.macro
    def achievement_badge(name: str, description: str, icon: str = "🏆", unlocked: bool = False) -> str:
        """Generate an achievement badge"""
        status_class = "unlocked" if unlocked else "locked"
        
        return f"""
        <div class="achievement-badge {status_class}">
            <div class="badge-icon">{icon}</div>
            <div class="badge-content">
                <h4 class="badge-name">{name}</h4>
                <p class="badge-description">{description}</p>
            </div>
            <div class="badge-status">
                {'✅ Unlocked' if unlocked else '🔒 Locked'}
            </div>
        </div>
        """
    
    @env.macro
    def phase_overview(phase_num: int, title: str, weeks: List[int], description: str) -> str:
        """Generate a phase overview component"""
        weeks_str = f"Weeks {weeks[0]}-{weeks[-1]}"
        
        return f"""
        <div class="phase-overview-card phase-{phase_num}">
            <div class="phase-header">
                <div class="phase-number">{phase_num}</div>
                <div class="phase-info">
                    <h3>{title}</h3>
                    <span class="phase-weeks">{weeks_str}</span>
                </div>
            </div>
            <div class="phase-description">
                <p>{description}</p>
            </div>
            <div class="phase-progress">
                <div class="progress-bar">
                    <div class="progress-fill phase-{phase_num}" style="width: 0%"></div>
                </div>
                <span class="progress-text">0/{len(weeks)} weeks completed</span>
            </div>
        </div>
        """
    
    # Environment variables
    env.variables['current_date'] = datetime.now().strftime("%Y-%m-%d")
    env.variables['current_year'] = datetime.now().year
    env.variables['total_weeks'] = 24
    env.variables['total_phases'] = 4
    
    # Study guide configuration
    env.variables['phases'] = {
        1: {"name": "Foundation LLM Architecture", "weeks": list(range(1, 7)), "color": "#2196f3"},
        2: {"name": "Advanced LLM Techniques", "weeks": list(range(7, 13)), "color": "#4caf50"},
        3: {"name": "LLM Agents and Tool Use", "weeks": list(range(13, 19)), "color": "#ff9800"},
        4: {"name": "Advanced Agent Architectures", "weeks": list(range(19, 25)), "color": "#9c27b0"}
    }
    
    # Healthcare focus areas
    env.variables['healthcare_areas'] = [
        "Clinical Decision Support",
        "Medical Text Processing", 
        "Safety & Compliance",
        "Multimodal Medical Data"
    ]
    
    # Apple Silicon features
    env.variables['apple_silicon_features'] = [
        "Metal Performance Shaders",
        "Unified Memory Architecture",
        "Energy Efficiency",
        "Performance Monitoring"
    ]
