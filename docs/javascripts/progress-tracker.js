// Enhanced Progress Tracking System for LLM Study Guide
// Supports persistent progress, study analytics, and gamification

class StudyProgressTracker {
  constructor() {
    this.storageKey = 'llm_study_progress';
    this.sessionKey = 'llm_study_session';
    this.notesKey = 'llm_study_notes';
    this.achievementsKey = 'llm_achievements';
    
    this.totalWeeks = 24;
    this.phases = {
      1: { weeks: [1,2,3,4,5,6], name: 'Foundation LLM Architecture' },
      2: { weeks: [7,8,9,10,11,12], name: 'Advanced LLM Techniques' },
      3: { weeks: [13,14,15,16,17,18], name: 'LLM Agents and Tool Use' },
      4: { weeks: [19,20,21,22,23,24], name: 'Advanced Agent Architectures' }
    };
    
    this.timer = {
      startTime: null,
      elapsedTime: 0,
      isRunning: false,
      interval: null
    };
    
    this.init();
  }
  
  init() {
    this.loadProgress();
    this.updateProgressDisplay();
    this.setupEventListeners();
    this.checkAchievements();
  }
  
  // Progress Management
  loadProgress() {
    const saved = localStorage.getItem(this.storageKey);
    this.progress = saved ? JSON.parse(saved) : this.createEmptyProgress();
  }
  
  createEmptyProgress() {
    const progress = {
      weeks: {},
      totalHours: 0,
      sessionsCompleted: 0,
      lastUpdated: new Date().toISOString(),
      achievements: []
    };
    
    for (let i = 1; i <= this.totalWeeks; i++) {
      progress.weeks[i] = {
        status: 'not-started', // not-started, in-progress, completed
        hoursSpent: 0,
        tasksCompleted: 0,
        totalTasks: 10, // Default, can be customized per week
        notes: '',
        completedDate: null
      };
    }
    
    return progress;
  }
  
  saveProgress() {
    this.progress.lastUpdated = new Date().toISOString();
    localStorage.setItem(this.storageKey, JSON.stringify(this.progress));
  }
  
  updateWeekProgress(week, updates) {
    if (!this.progress.weeks[week]) return;
    
    Object.assign(this.progress.weeks[week], updates);
    
    // Auto-update status based on tasks completed
    const weekData = this.progress.weeks[week];
    if (weekData.tasksCompleted === 0) {
      weekData.status = 'not-started';
    } else if (weekData.tasksCompleted < weekData.totalTasks) {
      weekData.status = 'in-progress';
    } else {
      weekData.status = 'completed';
      weekData.completedDate = new Date().toISOString();
    }
    
    this.saveProgress();
    this.updateProgressDisplay();
    this.checkAchievements();
  }
  
  // Display Updates
  updateProgressDisplay() {
    this.updateOverallProgress();
    this.updatePhaseProgress();
  }
  
  updateOverallProgress() {
    const completedWeeks = Object.values(this.progress.weeks)
      .filter(week => week.status === 'completed').length;
    const percentage = Math.round((completedWeeks / this.totalWeeks) * 100);
    
    const progressText = document.getElementById('overall-progress');
    const progressBar = document.querySelector('#overall-progress-bar .progress-fill');
    
    if (progressText) {
      progressText.textContent = `${percentage}% Complete (${completedWeeks}/${this.totalWeeks} weeks)`;
    }
    
    if (progressBar) {
      progressBar.style.width = `${percentage}%`;
    }
  }
  
  updatePhaseProgress() {
    Object.entries(this.phases).forEach(([phaseNum, phaseData]) => {
      const completedInPhase = phaseData.weeks.filter(week => 
        this.progress.weeks[week]?.status === 'completed'
      ).length;
      
      const percentage = Math.round((completedInPhase / phaseData.weeks.length) * 100);
      
      const phaseCard = document.querySelector(`[data-phase="${phaseNum}"]`);
      if (phaseCard) {
        const progressBar = phaseCard.querySelector('.progress-fill');
        const progressText = phaseCard.querySelector('.phase-percentage');
        
        if (progressBar) {
          progressBar.style.width = `${percentage}%`;
        }
        
        if (progressText) {
          progressText.textContent = `${completedInPhase}/${phaseData.weeks.length} weeks`;
        }
      }
    });
  }
  
  // Study Timer
  startTimer() {
    if (this.timer.isRunning) return;
    
    this.timer.startTime = Date.now() - this.timer.elapsedTime;
    this.timer.isRunning = true;
    
    this.timer.interval = setInterval(() => {
      this.updateTimerDisplay();
    }, 1000);
    
    document.getElementById('start-timer').disabled = true;
    document.getElementById('pause-timer').disabled = false;
    document.getElementById('stop-timer').disabled = false;
    
    this.saveSession();
  }
  
  pauseTimer() {
    if (!this.timer.isRunning) return;
    
    this.timer.isRunning = false;
    clearInterval(this.timer.interval);
    
    document.getElementById('start-timer').disabled = false;
    document.getElementById('pause-timer').disabled = true;
  }
  
  stopTimer() {
    this.timer.isRunning = false;
    clearInterval(this.timer.interval);
    
    const sessionMinutes = Math.round(this.timer.elapsedTime / 60000);
    const sessionHours = sessionMinutes / 60;
    
    // Update total hours
    this.progress.totalHours += sessionHours;
    this.progress.sessionsCompleted += 1;
    
    // Save session data
    this.saveSessionData(sessionMinutes);
    
    // Reset timer
    this.timer.elapsedTime = 0;
    this.updateTimerDisplay();
    
    document.getElementById('start-timer').disabled = false;
    document.getElementById('pause-timer').disabled = true;
    document.getElementById('stop-timer').disabled = true;
    
    this.saveProgress();
    this.showSessionSummary(sessionMinutes);
  }
  
  updateTimerDisplay() {
    if (this.timer.isRunning) {
      this.timer.elapsedTime = Date.now() - this.timer.startTime;
    }
    
    const totalSeconds = Math.floor(this.timer.elapsedTime / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    
    const display = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    
    const timerDisplay = document.getElementById('timer-display');
    if (timerDisplay) {
      timerDisplay.textContent = display;
    }
  }
  
  // Session Management
  saveSession() {
    const session = {
      topic: document.getElementById('study-topic')?.value || '',
      goal: document.getElementById('study-goal')?.value || '',
      startTime: new Date().toISOString()
    };
    
    localStorage.setItem(this.sessionKey, JSON.stringify(session));
  }
  
  saveSessionData(minutes) {
    const sessions = JSON.parse(localStorage.getItem('llm_study_sessions') || '[]');
    const session = JSON.parse(localStorage.getItem(this.sessionKey) || '{}');
    
    sessions.push({
      ...session,
      endTime: new Date().toISOString(),
      duration: minutes,
      notes: document.getElementById('quick-notes')?.value || ''
    });
    
    localStorage.setItem('llm_study_sessions', JSON.stringify(sessions));
    localStorage.removeItem(this.sessionKey);
  }
  
  showSessionSummary(minutes) {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    const timeStr = hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
    
    alert(`Study session completed!\nDuration: ${timeStr}\nTotal study time: ${Math.round(this.progress.totalHours * 10) / 10} hours`);
  }
  
  // Notes Management
  saveNotes() {
    const notes = document.getElementById('quick-notes')?.value || '';
    const timestamp = new Date().toISOString();
    
    const savedNotes = JSON.parse(localStorage.getItem(this.notesKey) || '[]');
    savedNotes.push({ content: notes, timestamp });
    
    localStorage.setItem(this.notesKey, JSON.stringify(savedNotes));
    
    alert('Notes saved successfully!');
  }
  
  clearNotes() {
    if (confirm('Are you sure you want to clear your notes?')) {
      document.getElementById('quick-notes').value = '';
    }
  }
  
  exportNotes() {
    const notes = JSON.parse(localStorage.getItem(this.notesKey) || '[]');
    const content = notes.map(note => 
      `${new Date(note.timestamp).toLocaleString()}\n${note.content}\n\n`
    ).join('');
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'llm-study-notes.txt';
    a.click();
    URL.revokeObjectURL(url);
  }
  
  // Achievement System
  checkAchievements() {
    const achievements = [
      {
        id: 'first_week',
        name: 'First Steps',
        description: 'Complete your first week of study',
        condition: () => Object.values(this.progress.weeks).some(w => w.status === 'completed'),
        icon: '🎯'
      },
      {
        id: 'phase_1_complete',
        name: 'Foundation Master',
        description: 'Complete Phase 1: Foundation LLM Architecture',
        condition: () => this.phases[1].weeks.every(w => this.progress.weeks[w].status === 'completed'),
        icon: '🏗️'
      },
      {
        id: 'study_streak_7',
        name: 'Week Warrior',
        description: 'Study for 7 consecutive days',
        condition: () => this.checkStudyStreak(7),
        icon: '🔥'
      },
      {
        id: 'total_hours_50',
        name: 'Dedicated Learner',
        description: 'Accumulate 50 hours of study time',
        condition: () => this.progress.totalHours >= 50,
        icon: '⏰'
      }
    ];
    
    achievements.forEach(achievement => {
      if (!this.progress.achievements.includes(achievement.id) && achievement.condition()) {
        this.unlockAchievement(achievement);
      }
    });
  }
  
  unlockAchievement(achievement) {
    this.progress.achievements.push(achievement.id);
    this.saveProgress();
    this.showAchievementNotification(achievement);
  }
  
  showAchievementNotification(achievement) {
    const notification = document.createElement('div');
    notification.className = 'achievement-notification';
    notification.innerHTML = `
      <div class="achievement-content">
        <div class="achievement-icon">${achievement.icon}</div>
        <div class="achievement-text">
          <div class="achievement-title">Achievement Unlocked!</div>
          <div class="achievement-name">${achievement.name}</div>
          <div class="achievement-desc">${achievement.description}</div>
        </div>
      </div>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.classList.add('show');
    }, 100);
    
    setTimeout(() => {
      notification.classList.remove('show');
      setTimeout(() => {
        document.body.removeChild(notification);
      }, 300);
    }, 5000);
  }
  
  checkStudyStreak(days) {
    // Implementation for checking study streaks
    // This would require tracking daily study sessions
    return false; // Placeholder
  }
  
  // Event Listeners
  setupEventListeners() {
    // Modal controls
    window.openStudyTimer = () => {
      document.getElementById('study-timer-modal').style.display = 'block';
    };
    
    window.closeStudyTimer = () => {
      document.getElementById('study-timer-modal').style.display = 'none';
    };
    
    window.openNotepad = () => {
      document.getElementById('notes-modal').style.display = 'block';
    };
    
    window.closeNotepad = () => {
      document.getElementById('notes-modal').style.display = 'none';
    };
    
    window.viewAchievements = () => {
      this.showAchievements();
    };
    
    // Timer controls
    window.startTimer = () => this.startTimer();
    window.pauseTimer = () => this.pauseTimer();
    window.stopTimer = () => this.stopTimer();
    
    // Notes controls
    window.saveNotes = () => this.saveNotes();
    window.clearNotes = () => this.clearNotes();
    window.exportNotes = () => this.exportNotes();
    
    // Close modals when clicking outside
    window.onclick = (event) => {
      const modals = document.querySelectorAll('.modal');
      modals.forEach(modal => {
        if (event.target === modal) {
          modal.style.display = 'none';
        }
      });
    };
  }
  
  showAchievements() {
    // Implementation for showing achievements gallery
    alert('Achievements gallery coming soon!');
  }
}

// Initialize the progress tracker when the page loads
document.addEventListener('DOMContentLoaded', () => {
  window.studyTracker = new StudyProgressTracker();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = StudyProgressTracker;
}
