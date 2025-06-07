// Study Components - Interactive elements for the LLM study guide
// Provides reusable components for checkpoints, progress tracking, and study aids

class StudyComponents {
  constructor() {
    this.init();
  }
  
  init() {
    this.setupCheckpoints();
    this.setupInteractiveProgress();
    this.setupStudyNotes();
    this.setupCodeExamples();
    this.setupQuickActions();
  }
  
  // Study Checkpoints
  setupCheckpoints() {
    document.addEventListener('DOMContentLoaded', () => {
      this.createCheckpoints();
    });
  }
  
  createCheckpoints() {
    // Find all h3 and h4 headings that could be checkpoints
    const headings = document.querySelectorAll('h3, h4');
    
    headings.forEach((heading, index) => {
      if (this.shouldCreateCheckpoint(heading)) {
        this.addCheckpointToHeading(heading, index);
      }
    });
  }
  
  shouldCreateCheckpoint(heading) {
    const text = heading.textContent.toLowerCase();
    const checkpointKeywords = [
      'mathematical foundations',
      'key readings',
      'healthcare applications',
      'hands-on deliverable',
      'implementation',
      'exercise',
      'practice'
    ];
    
    return checkpointKeywords.some(keyword => text.includes(keyword));
  }
  
  addCheckpointToHeading(heading, index) {
    const checkpointId = `checkpoint-${index}`;
    const isCompleted = this.getCheckpointStatus(checkpointId);
    
    const checkpoint = document.createElement('div');
    checkpoint.className = 'study-checkpoint';
    checkpoint.innerHTML = `
      <div class="checkpoint-content">
        <h4>${heading.textContent}</h4>
        <div class="checkpoint-actions">
          <button class="checkpoint-btn ${isCompleted ? 'completed' : ''}" 
                  onclick="studyComponents.toggleCheckpoint('${checkpointId}', this)">
            ${isCompleted ? '✅ Completed' : '⏳ Mark Complete'}
          </button>
          <button class="checkpoint-btn" onclick="studyComponents.addCheckpointNote('${checkpointId}')">
            📝 Add Note
          </button>
        </div>
      </div>
    `;
    
    heading.parentNode.insertBefore(checkpoint, heading.nextSibling);
    heading.style.display = 'none'; // Hide original heading
  }
  
  toggleCheckpoint(checkpointId, button) {
    const isCompleted = button.classList.contains('completed');
    
    if (isCompleted) {
      button.classList.remove('completed');
      button.textContent = '⏳ Mark Complete';
      this.saveCheckpointStatus(checkpointId, false);
    } else {
      button.classList.add('completed');
      button.textContent = '✅ Completed';
      this.saveCheckpointStatus(checkpointId, true);
      this.showCompletionAnimation(button);
    }
  }
  
  saveCheckpointStatus(checkpointId, completed) {
    const checkpoints = JSON.parse(localStorage.getItem('study_checkpoints') || '{}');
    checkpoints[checkpointId] = completed;
    localStorage.setItem('study_checkpoints', JSON.stringify(checkpoints));
  }
  
  getCheckpointStatus(checkpointId) {
    const checkpoints = JSON.parse(localStorage.getItem('study_checkpoints') || '{}');
    return checkpoints[checkpointId] || false;
  }
  
  showCompletionAnimation(button) {
    button.style.transform = 'scale(1.1)';
    setTimeout(() => {
      button.style.transform = 'scale(1)';
    }, 200);
  }
  
  addCheckpointNote(checkpointId) {
    const note = prompt('Add a note for this checkpoint:');
    if (note) {
      const notes = JSON.parse(localStorage.getItem('checkpoint_notes') || '{}');
      notes[checkpointId] = note;
      localStorage.setItem('checkpoint_notes', JSON.stringify(notes));
      alert('Note saved!');
    }
  }
  
  // Interactive Progress Bars
  setupInteractiveProgress() {
    document.addEventListener('DOMContentLoaded', () => {
      this.createInteractiveProgress();
    });
  }
  
  createInteractiveProgress() {
    // Look for tables that could be converted to interactive progress
    const tables = document.querySelectorAll('table');
    
    tables.forEach(table => {
      if (this.isProgressTable(table)) {
        this.convertToInteractiveProgress(table);
      }
    });
  }
  
  isProgressTable(table) {
    const headers = table.querySelectorAll('th');
    const headerText = Array.from(headers).map(h => h.textContent.toLowerCase());
    
    return headerText.some(text => 
      text.includes('status') || 
      text.includes('progress') || 
      text.includes('completion')
    );
  }
  
  convertToInteractiveProgress(table) {
    const rows = table.querySelectorAll('tbody tr');
    const progressContainer = document.createElement('div');
    progressContainer.className = 'interactive-progress-container';
    
    rows.forEach((row, index) => {
      const cells = row.querySelectorAll('td');
      if (cells.length >= 2) {
        const label = cells[0].textContent.trim();
        const status = cells[1].textContent.trim();
        
        const progressItem = this.createProgressItem(label, status, index);
        progressContainer.appendChild(progressItem);
      }
    });
    
    table.parentNode.insertBefore(progressContainer, table);
    table.style.display = 'none';
  }
  
  createProgressItem(label, status, index) {
    const percentage = this.statusToPercentage(status);
    const itemId = `progress-item-${index}`;
    
    const item = document.createElement('div');
    item.className = 'interactive-progress';
    item.innerHTML = `
      <div class="progress-header">
        <span class="progress-label">${label}</span>
        <span class="progress-percentage">${percentage}%</span>
      </div>
      <div class="progress-bar-interactive">
        <div class="progress-fill-interactive" style="width: ${percentage}%"></div>
      </div>
    `;
    
    item.addEventListener('click', () => {
      this.updateProgressItem(itemId, item);
    });
    
    return item;
  }
  
  statusToPercentage(status) {
    const statusMap = {
      'not started': 0,
      'pending': 0,
      'in progress': 50,
      'completed': 100,
      'done': 100
    };
    
    const normalizedStatus = status.toLowerCase();
    return statusMap[normalizedStatus] || 0;
  }
  
  updateProgressItem(itemId, element) {
    const currentPercentage = parseInt(element.querySelector('.progress-percentage').textContent);
    const newPercentage = currentPercentage >= 100 ? 0 : currentPercentage + 25;
    
    element.querySelector('.progress-percentage').textContent = `${newPercentage}%`;
    element.querySelector('.progress-fill-interactive').style.width = `${newPercentage}%`;
    
    // Save progress
    const progress = JSON.parse(localStorage.getItem('interactive_progress') || '{}');
    progress[itemId] = newPercentage;
    localStorage.setItem('interactive_progress', JSON.stringify(progress));
  }
  
  // Study Notes Widget
  setupStudyNotes() {
    document.addEventListener('DOMContentLoaded', () => {
      this.addStudyNotesWidgets();
    });
  }
  
  addStudyNotesWidgets() {
    // Add notes widgets after major sections
    const sections = document.querySelectorAll('h2, h3');
    
    sections.forEach((section, index) => {
      if (index % 3 === 0) { // Add widget every 3rd section
        this.addNotesWidget(section, index);
      }
    });
  }
  
  addNotesWidget(section, index) {
    const widgetId = `notes-widget-${index}`;
    const savedNote = this.getSavedNote(widgetId);
    
    const widget = document.createElement('div');
    widget.className = 'study-notes-widget';
    widget.innerHTML = `
      <div class="notes-header">
        <h4 class="notes-title">📝 Study Notes</h4>
        <button class="notes-toggle" onclick="studyComponents.toggleNotes('${widgetId}')">
          ${savedNote ? 'Show Notes' : 'Add Notes'}
        </button>
      </div>
      <div class="notes-content" id="${widgetId}">
        <textarea class="notes-textarea" placeholder="Add your notes, questions, or insights here...">${savedNote}</textarea>
        <button onclick="studyComponents.saveNote('${widgetId}')" style="margin-top: 0.5rem; padding: 0.5rem 1rem; background: var(--md-primary-fg-color); color: var(--md-primary-bg-color); border: none; border-radius: 4px; cursor: pointer;">
          Save Note
        </button>
      </div>
    `;
    
    section.parentNode.insertBefore(widget, section.nextSibling);
  }
  
  toggleNotes(widgetId) {
    const content = document.getElementById(widgetId);
    const button = content.previousElementSibling.querySelector('.notes-toggle');
    
    if (content.classList.contains('expanded')) {
      content.classList.remove('expanded');
      button.textContent = 'Show Notes';
    } else {
      content.classList.add('expanded');
      button.textContent = 'Hide Notes';
    }
  }
  
  saveNote(widgetId) {
    const textarea = document.querySelector(`#${widgetId} textarea`);
    const note = textarea.value;
    
    const notes = JSON.parse(localStorage.getItem('section_notes') || '{}');
    notes[widgetId] = note;
    localStorage.setItem('section_notes', JSON.stringify(notes));
    
    alert('Note saved!');
  }
  
  getSavedNote(widgetId) {
    const notes = JSON.parse(localStorage.getItem('section_notes') || '{}');
    return notes[widgetId] || '';
  }
  
  // Enhanced Code Examples
  setupCodeExamples() {
    document.addEventListener('DOMContentLoaded', () => {
      this.enhanceCodeBlocks();
    });
  }
  
  enhanceCodeBlocks() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach((block, index) => {
      this.addCodeEnhancements(block, index);
    });
  }
  
  addCodeEnhancements(codeBlock, index) {
    const pre = codeBlock.parentElement;
    const wrapper = document.createElement('div');
    wrapper.className = 'enhanced-code-block';
    
    // Add header with language and actions
    const header = document.createElement('div');
    header.className = 'code-header';
    header.innerHTML = `
      <span class="code-language">${this.detectLanguage(codeBlock)}</span>
      <div class="code-actions">
        <button onclick="studyComponents.copyCode(${index})" title="Copy code">📋</button>
        <button onclick="studyComponents.runCode(${index})" title="Run code">▶️</button>
        <button onclick="studyComponents.explainCode(${index})" title="Explain code">💡</button>
      </div>
    `;
    
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(header);
    wrapper.appendChild(pre);
    
    // Add line numbers
    this.addLineNumbers(codeBlock);
  }
  
  detectLanguage(codeBlock) {
    const className = codeBlock.className;
    const match = className.match(/language-(\w+)/);
    return match ? match[1].toUpperCase() : 'CODE';
  }
  
  addLineNumbers(codeBlock) {
    const lines = codeBlock.textContent.split('\n');
    const lineNumbers = document.createElement('div');
    lineNumbers.className = 'line-numbers';
    
    lines.forEach((_, index) => {
      const lineNumber = document.createElement('span');
      lineNumber.textContent = index + 1;
      lineNumbers.appendChild(lineNumber);
    });
    
    codeBlock.parentElement.insertBefore(lineNumbers, codeBlock);
  }
  
  copyCode(index) {
    const codeBlocks = document.querySelectorAll('pre code');
    const code = codeBlocks[index].textContent;
    
    navigator.clipboard.writeText(code).then(() => {
      this.showToast('Code copied to clipboard!');
    });
  }
  
  runCode(index) {
    // Placeholder for code execution
    this.showToast('Code execution feature coming soon!');
  }
  
  explainCode(index) {
    // Placeholder for code explanation
    this.showToast('Code explanation feature coming soon!');
  }
  
  // Quick Actions
  setupQuickActions() {
    this.addFloatingActionButton();
  }
  
  addFloatingActionButton() {
    const fab = document.createElement('div');
    fab.className = 'floating-action-button';
    fab.innerHTML = `
      <button class="fab-main" onclick="studyComponents.toggleFabMenu()">⚡</button>
      <div class="fab-menu" id="fab-menu">
        <button onclick="studyComponents.quickBookmark()" title="Bookmark this page">🔖</button>
        <button onclick="studyComponents.quickNote()" title="Quick note">📝</button>
        <button onclick="studyComponents.quickTimer()" title="Start timer">⏱️</button>
        <button onclick="studyComponents.quickProgress()" title="Update progress">📊</button>
      </div>
    `;
    
    document.body.appendChild(fab);
  }
  
  toggleFabMenu() {
    const menu = document.getElementById('fab-menu');
    menu.classList.toggle('open');
  }
  
  quickBookmark() {
    const bookmarks = JSON.parse(localStorage.getItem('study_bookmarks') || '[]');
    const bookmark = {
      url: window.location.href,
      title: document.title,
      timestamp: new Date().toISOString()
    };
    
    bookmarks.push(bookmark);
    localStorage.setItem('study_bookmarks', JSON.stringify(bookmarks));
    this.showToast('Page bookmarked!');
  }
  
  quickNote() {
    const note = prompt('Quick note:');
    if (note) {
      const notes = JSON.parse(localStorage.getItem('quick_notes') || '[]');
      notes.push({
        content: note,
        url: window.location.href,
        timestamp: new Date().toISOString()
      });
      localStorage.setItem('quick_notes', JSON.stringify(notes));
      this.showToast('Note saved!');
    }
  }
  
  quickTimer() {
    if (window.studyTracker) {
      window.studyTracker.startTimer();
    }
  }
  
  quickProgress() {
    // Open progress modal or update current page progress
    this.showToast('Progress update feature coming soon!');
  }
  
  // Utility Methods
  showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    toast.textContent = message;
    toast.style.cssText = `
      position: fixed;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: var(--md-primary-fg-color);
      color: var(--md-primary-bg-color);
      padding: 0.75rem 1.5rem;
      border-radius: 25px;
      z-index: 1000;
      font-size: 0.9rem;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
      document.body.removeChild(toast);
    }, 3000);
  }
}

// Initialize study components
document.addEventListener('DOMContentLoaded', () => {
  window.studyComponents = new StudyComponents();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = StudyComponents;
}
