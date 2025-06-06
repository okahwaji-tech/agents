// MathJax configuration for LLM & Agentic Workflows documentation

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {
      '[+]': ['ams', 'newcommand', 'configmacros']
    },
    macros: {
      // Common mathematical notation for LLMs
      "softmax": "\\text{softmax}",
      "argmax": "\\text{argmax}",
      "argmin": "\\text{argmin}",
      "log": "\\text{log}",
      "exp": "\\text{exp}",
      "sigmoid": "\\text{sigmoid}",
      "tanh": "\\text{tanh}",
      "relu": "\\text{ReLU}",
      "gelu": "\\text{GELU}",
      
      // Probability and statistics
      "prob": "\\mathbb{P}",
      "expect": "\\mathbb{E}",
      "var": "\\text{Var}",
      "cov": "\\text{Cov}",
      "given": "\\mid",
      
      // Linear algebra
      "transpose": "^\\top",
      "inverse": "^{-1}",
      "trace": "\\text{tr}",
      "rank": "\\text{rank}",
      "det": "\\text{det}",
      "diag": "\\text{diag}",
      
      // Information theory
      "entropy": "H",
      "kldiv": "D_{\\text{KL}}",
      "mutualinfo": "I",
      "crossentropy": "H_{\\text{cross}}",
      
      // Neural network notation
      "weights": "\\mathbf{W}",
      "bias": "\\mathbf{b}",
      "input": "\\mathbf{x}",
      "output": "\\mathbf{y}",
      "hidden": "\\mathbf{h}",
      "embedding": "\\mathbf{e}",
      
      // Attention mechanism notation
      "query": "\\mathbf{Q}",
      "key": "\\mathbf{K}",
      "value": "\\mathbf{V}",
      "attention": "\\text{Attention}",
      "multihead": "\\text{MultiHead}",
      
      // Transformer notation
      "layernorm": "\\text{LayerNorm}",
      "feedforward": "\\text{FFN}",
      "positional": "\\text{PE}",
      
      // Optimization
      "gradient": "\\nabla",
      "loss": "\\mathcal{L}",
      "objective": "\\mathcal{J}",
      "regularizer": "\\mathcal{R}",
      
      // Reinforcement learning
      "state": "s",
      "action": "a",
      "reward": "r",
      "policy": "\\pi",
      "qvalue": "Q",
      "vvalue": "V",
      "advantage": "A",
      
      // Sets and spaces
      "reals": "\\mathbb{R}",
      "integers": "\\mathbb{Z}",
      "naturals": "\\mathbb{N}",
      "vocab": "\\mathcal{V}",
      "dataset": "\\mathcal{D}",
      "model": "\\mathcal{M}",
      
      // Common functions
      "norm": "\\|#1\\|",
      "abs": "\\left|#1\\right|",
      "floor": "\\lfloor#1\\rfloor",
      "ceil": "\\lceil#1\\rceil",
      
      // Healthcare-specific notation
      "patient": "p",
      "symptom": "s",
      "disease": "d",
      "treatment": "t",
      "outcome": "o",
      "clinical": "\\text{clinical}",
      "medical": "\\text{medical}"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  loader: {
    load: ['[tex]/ams', '[tex]/newcommand', '[tex]/configmacros']
  }
};

// Custom styling for mathematical expressions
document.addEventListener('DOMContentLoaded', function() {
  // Add custom classes to math expressions for styling
  const mathElements = document.querySelectorAll('.MathJax');
  mathElements.forEach(function(element) {
    element.classList.add('math-expression');
  });
  
  // Add click-to-copy functionality for mathematical expressions
  const mathExpressions = document.querySelectorAll('.arithmatex');
  mathExpressions.forEach(function(expr) {
    expr.addEventListener('click', function() {
      const mathText = expr.textContent || expr.innerText;
      navigator.clipboard.writeText(mathText).then(function() {
        // Show temporary tooltip
        const tooltip = document.createElement('div');
        tooltip.textContent = 'Copied!';
        tooltip.style.cssText = `
          position: absolute;
          background: #333;
          color: white;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          z-index: 1000;
          pointer-events: none;
        `;
        
        const rect = expr.getBoundingClientRect();
        tooltip.style.left = rect.left + 'px';
        tooltip.style.top = (rect.top - 30) + 'px';
        
        document.body.appendChild(tooltip);
        
        setTimeout(function() {
          document.body.removeChild(tooltip);
        }, 1000);
      });
    });
    
    // Add hover effect
    expr.style.cursor = 'pointer';
    expr.title = 'Click to copy';
  });
});

// Progress tracking functionality
function updateProgress(weekId, taskId, status) {
  const progressKey = `progress_${weekId}_${taskId}`;
  localStorage.setItem(progressKey, status);
  
  // Update UI
  const element = document.getElementById(`${weekId}_${taskId}`);
  if (element) {
    element.className = `status-${status}`;
    element.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    
    // Add animation
    element.classList.add('progress-updated');
    setTimeout(() => {
      element.classList.remove('progress-updated');
    }, 500);
  }
  
  // Update overall progress
  updateOverallProgress(weekId);
}

function updateOverallProgress(weekId) {
  const tasks = document.querySelectorAll(`[id^="${weekId}_"]`);
  const completed = Array.from(tasks).filter(task => 
    task.classList.contains('status-completed')
  ).length;
  
  const progressBar = document.getElementById(`progress_${weekId}`);
  if (progressBar) {
    const percentage = (completed / tasks.length) * 100;
    progressBar.style.width = `${percentage}%`;
  }
}

// Load saved progress on page load
document.addEventListener('DOMContentLoaded', function() {
  // Load progress from localStorage
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key.startsWith('progress_')) {
      const status = localStorage.getItem(key);
      const [, weekId, taskId] = key.split('_');
      const element = document.getElementById(`${weekId}_${taskId}`);
      if (element) {
        element.className = `status-${status}`;
        element.textContent = status.charAt(0).toUpperCase() + status.slice(1);
      }
    }
  }
  
  // Update all progress bars
  const weeks = ['week1', 'week2', 'week3', 'week4', 'week5', 'week6'];
  weeks.forEach(updateOverallProgress);
});

// Code copy functionality
function addCodeCopyButtons() {
  const codeBlocks = document.querySelectorAll('pre code');
  codeBlocks.forEach(function(codeBlock) {
    const pre = codeBlock.parentElement;
    const button = document.createElement('button');
    button.textContent = 'Copy';
    button.className = 'copy-button';
    button.style.cssText = `
      position: absolute;
      top: 8px;
      right: 8px;
      background: #333;
      color: white;
      border: none;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 12px;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.3s;
    `;
    
    pre.style.position = 'relative';
    pre.appendChild(button);
    
    pre.addEventListener('mouseenter', () => {
      button.style.opacity = '1';
    });
    
    pre.addEventListener('mouseleave', () => {
      button.style.opacity = '0';
    });
    
    button.addEventListener('click', function() {
      navigator.clipboard.writeText(codeBlock.textContent).then(function() {
        button.textContent = 'Copied!';
        setTimeout(function() {
          button.textContent = 'Copy';
        }, 1000);
      });
    });
  });
}

// Initialize code copy buttons when page loads
document.addEventListener('DOMContentLoaded', addCodeCopyButtons);

// Apple Silicon performance monitoring
function monitorAppleSiliconPerformance() {
  if ('performance' in window && 'memory' in performance) {
    const memoryInfo = performance.memory;
    const memoryUsage = {
      used: Math.round(memoryInfo.usedJSHeapSize / 1048576), // MB
      total: Math.round(memoryInfo.totalJSHeapSize / 1048576), // MB
      limit: Math.round(memoryInfo.jsHeapSizeLimit / 1048576) // MB
    };
    
    console.log('Memory Usage:', memoryUsage);
    
    // Show performance indicator if memory usage is high
    if (memoryUsage.used / memoryUsage.limit > 0.8) {
      showPerformanceWarning();
    }
  }
}

function showPerformanceWarning() {
  const warning = document.createElement('div');
  warning.innerHTML = `
    <div style="
      position: fixed;
      top: 20px;
      right: 20px;
      background: #ff9800;
      color: white;
      padding: 12px;
      border-radius: 8px;
      z-index: 1000;
      max-width: 300px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    ">
      <strong>🍎 Apple Silicon Optimization</strong><br>
      High memory usage detected. Consider closing other applications for optimal performance.
      <button onclick="this.parentElement.parentElement.remove()" style="
        float: right;
        background: none;
        border: none;
        color: white;
        font-size: 16px;
        cursor: pointer;
        margin-left: 8px;
      ">×</button>
    </div>
  `;
  document.body.appendChild(warning);
  
  // Auto-remove after 10 seconds
  setTimeout(() => {
    if (warning.parentElement) {
      warning.remove();
    }
  }, 10000);
}

// Monitor performance periodically
setInterval(monitorAppleSiliconPerformance, 30000); // Every 30 seconds
