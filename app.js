// Sample data from the provided JSON
const sampleData = {
  samplePosts: [
    {"text": "I'm absolutely amazed by the latest AI breakthrough! This technology is revolutionary.", "sentiment": "Positive", "confidence": 0.94, "platform": "Twitter", "timestamp": "2025-08-20T14:30:00Z", "engagement": 245},
    {"text": "This AI system is terrible and keeps making mistakes. Very disappointed.", "sentiment": "Negative", "confidence": 0.89, "platform": "Reddit", "timestamp": "2025-08-20T12:15:00Z", "engagement": 67},
    {"text": "The machine learning algorithm performs as expected. Standard results.", "sentiment": "Neutral", "confidence": 0.72, "platform": "Twitter", "timestamp": "2025-08-20T10:45:00Z", "engagement": 23},
    {"text": "Incredible advances in natural language processing! The future is here.", "sentiment": "Positive", "confidence": 0.96, "platform": "Reddit", "timestamp": "2025-08-19T16:20:00Z", "engagement": 156},
    {"text": "AI replacing human jobs is a serious concern that needs addressing.", "sentiment": "Negative", "confidence": 0.85, "platform": "Twitter", "timestamp": "2025-08-19T09:30:00Z", "engagement": 89},
    {"text": "The neural network architecture looks interesting from a technical perspective.", "sentiment": "Neutral", "confidence": 0.68, "platform": "Reddit", "timestamp": "2025-08-18T20:15:00Z", "engagement": 34},
    {"text": "Love how AI is transforming healthcare and saving lives!", "sentiment": "Positive", "confidence": 0.91, "platform": "Twitter", "timestamp": "2025-08-18T11:45:00Z", "engagement": 312},
    {"text": "These chatbots are frustrating and don't understand context properly.", "sentiment": "Negative", "confidence": 0.87, "platform": "Reddit", "timestamp": "2025-08-17T14:20:00Z", "engagement": 45},
    {"text": "AI research funding increased by 15% according to the latest report.", "sentiment": "Neutral", "confidence": 0.75, "platform": "Twitter", "timestamp": "2025-08-17T08:30:00Z", "engagement": 78},
    {"text": "The computer vision model exceeded all expectations! Truly impressive work.", "sentiment": "Positive", "confidence": 0.93, "platform": "Reddit", "timestamp": "2025-08-16T19:10:00Z", "engagement": 203}
  ],
  topWords: {
    positive: ["amazing", "incredible", "revolutionary", "breakthrough", "excellent", "fantastic", "innovative", "impressive", "outstanding", "brilliant"],
    negative: ["terrible", "disappointing", "frustrating", "concerning", "awful", "problematic", "unreliable", "biased", "dangerous", "lacking"],
    neutral: ["standard", "typical", "normal", "expected", "regular", "average", "common", "usual", "conventional", "moderate"]
  },
  trendData: [
    {"date": "2025-08-15", "positive": 45, "negative": 25, "neutral": 30},
    {"date": "2025-08-16", "positive": 43, "negative": 29, "neutral": 28},
    {"date": "2025-08-17", "positive": 41, "negative": 31, "neutral": 28},
    {"date": "2025-08-18", "positive": 44, "negative": 27, "neutral": 29},
    {"date": "2025-08-19", "positive": 42, "negative": 30, "neutral": 28},
    {"date": "2025-08-20", "positive": 40, "negative": 32, "neutral": 28},
    {"date": "2025-08-21", "positive": 43, "negative": 28, "neutral": 29},
    {"date": "2025-08-22", "positive": 45, "negative": 26, "neutral": 29}
  ]
};

// Extended sample posts for different sentiments
const extendedPosts = {
  positive: [
    {"text": "I'm absolutely amazed by the latest AI breakthrough! This technology is revolutionary.", "confidence": 0.94, "platform": "Twitter", "timestamp": "2025-08-20T14:30:00Z", "engagement": 245},
    {"text": "Incredible advances in natural language processing! The future is here.", "confidence": 0.96, "platform": "Reddit", "timestamp": "2025-08-19T16:20:00Z", "engagement": 156},
    {"text": "Love how AI is transforming healthcare and saving lives!", "confidence": 0.91, "platform": "Twitter", "timestamp": "2025-08-18T11:45:00Z", "engagement": 312},
    {"text": "The computer vision model exceeded all expectations! Truly impressive work.", "confidence": 0.93, "platform": "Reddit", "timestamp": "2025-08-16T19:10:00Z", "engagement": 203},
    {"text": "This machine learning breakthrough is going to change everything for the better!", "confidence": 0.89, "platform": "Twitter", "timestamp": "2025-08-21T09:15:00Z", "engagement": 178}
  ],
  negative: [
    {"text": "This AI system is terrible and keeps making mistakes. Very disappointed.", "confidence": 0.89, "platform": "Reddit", "timestamp": "2025-08-20T12:15:00Z", "engagement": 67},
    {"text": "AI replacing human jobs is a serious concern that needs addressing.", "confidence": 0.85, "platform": "Twitter", "timestamp": "2025-08-19T09:30:00Z", "engagement": 89},
    {"text": "These chatbots are frustrating and don't understand context properly.", "confidence": 0.87, "platform": "Reddit", "timestamp": "2025-08-17T14:20:00Z", "engagement": 45},
    {"text": "The AI bias in these systems is really concerning and problematic.", "confidence": 0.92, "platform": "Twitter", "timestamp": "2025-08-16T16:45:00Z", "engagement": 134},
    {"text": "Another AI failure that shows we're not ready for this technology yet.", "confidence": 0.88, "platform": "Reddit", "timestamp": "2025-08-21T11:20:00Z", "engagement": 76}
  ],
  neutral: [
    {"text": "The machine learning algorithm performs as expected. Standard results.", "confidence": 0.72, "platform": "Twitter", "timestamp": "2025-08-20T10:45:00Z", "engagement": 23},
    {"text": "The neural network architecture looks interesting from a technical perspective.", "confidence": 0.68, "platform": "Reddit", "timestamp": "2025-08-18T20:15:00Z", "engagement": 34},
    {"text": "AI research funding increased by 15% according to the latest report.", "confidence": 0.75, "platform": "Twitter", "timestamp": "2025-08-17T08:30:00Z", "engagement": 78},
    {"text": "The new AI model shows moderate improvements over previous versions.", "confidence": 0.71, "platform": "Reddit", "timestamp": "2025-08-16T13:25:00Z", "engagement": 52},
    {"text": "Standard implementation of the transformer architecture with expected results.", "confidence": 0.69, "platform": "Twitter", "timestamp": "2025-08-21T15:30:00Z", "engagement": 41}
  ]
};

// Chart instances
let sentimentChart = null;
let trendsChart = null;
let platformChart = null;

// DOM Elements
const elements = {
  keyword: document.getElementById('keyword'),
  platform: document.getElementById('platform'),
  postCount: document.getElementById('postCount'),
  postCountValue: document.getElementById('postCountValue'),
  dateFrom: document.getElementById('dateFrom'),
  dateTo: document.getElementById('dateTo'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  loadingState: document.getElementById('loadingState'),
  dashboardContent: document.getElementById('dashboardContent'),
  currentTopic: document.getElementById('currentTopic'),
  currentPosts: document.getElementById('currentPosts'),
  currentSources: document.getElementById('currentSources'),
  currentPeriod: document.getElementById('currentPeriod')
};

// Global variables
let currentData = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
  createMobileMenuToggle();
  initializeEventListeners();
  updateAnalysisSummary();
  initializeCharts();
  updateWordAnalysis();
  initializePostTabs();
  updatePostsDisplay();
  
  // Ensure sidebar is visible initially
  const sidebar = document.querySelector('.sidebar');
  if (window.innerWidth > 1024) {
    sidebar.classList.remove('open');
  }
});

// Create mobile menu toggle button
function createMobileMenuToggle() {
  const toggleButton = document.createElement('button');
  toggleButton.className = 'mobile-menu-toggle';
  toggleButton.innerHTML = '☰ Menu';
  toggleButton.addEventListener('click', toggleSidebar);
  document.body.appendChild(toggleButton);
}

// Toggle sidebar visibility
function toggleSidebar() {
  const sidebar = document.querySelector('.sidebar');
  sidebar.classList.toggle('open');
}

// Event Listeners
function initializeEventListeners() {
  // Range slider update
  elements.postCount.addEventListener('input', function() {
    elements.postCountValue.textContent = parseInt(this.value).toLocaleString();
    updateAnalysisSummary();
  });

  // Form field updates
  [elements.keyword, elements.platform, elements.dateFrom, elements.dateTo].forEach(element => {
    element.addEventListener('change', updateAnalysisSummary);
  });

  // Analyze button
  elements.analyzeBtn.addEventListener('click', performAnalysis);

  // Tab buttons
  document.querySelectorAll('.tab-button').forEach(button => {
    button.addEventListener('click', function() {
      const tabName = this.dataset.tab;
      switchTab(tabName);
    });
  });

  // Export buttons
  document.querySelectorAll('.export-buttons .btn').forEach(button => {
    button.addEventListener('click', function() {
      const text = this.textContent.trim();
      if (text.includes('Download Dataset')) {
        simulateDownload('sentiment_data.csv');
      } else if (text.includes('Generate Report')) {
        simulateDownload('sentiment_report.pdf');
      } else if (text.includes('Share Dashboard')) {
        copyShareLink();
      } else if (text.includes('Print View')) {
        window.print();
      }
    });
  });
  
  // Close sidebar when clicking outside on mobile
  document.addEventListener('click', function(e) {
    const sidebar = document.querySelector('.sidebar');
    const toggleButton = document.querySelector('.mobile-menu-toggle');
    
    if (window.innerWidth <= 1024 && 
        !sidebar.contains(e.target) && 
        !toggleButton.contains(e.target) && 
        sidebar.classList.contains('open')) {
      sidebar.classList.remove('open');
    }
  });
}

// Update analysis summary
function updateAnalysisSummary() {
  const keyword = elements.keyword.value || 'artificial intelligence';
  const postCount = parseInt(elements.postCount.value).toLocaleString();
  const platform = elements.platform.value;
  const dateFrom = elements.dateFrom.value;
  const dateTo = elements.dateTo.value;

  elements.currentTopic.textContent = keyword;
  elements.currentPosts.textContent = postCount;
  
  const platformText = {
    'both': 'Twitter + Reddit',
    'twitter': 'Twitter',
    'reddit': 'Reddit'
  };
  elements.currentSources.textContent = platformText[platform];

  const fromDate = new Date(dateFrom).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  const toDate = new Date(dateTo).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  elements.currentPeriod.textContent = `${fromDate} - ${toDate}`;
}

// Perform analysis with loading state
async function performAnalysis() {
  // Show loading state
  elements.analyzeBtn.classList.add('loading');
  elements.loadingState.classList.remove('hidden');
  elements.dashboardContent.classList.add('hidden');

  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 3000));

  // Generate new data based on current settings
  generateAnalysisData();

  // Hide loading state
  elements.analyzeBtn.classList.remove('loading');
  elements.loadingState.classList.add('hidden');
  elements.dashboardContent.classList.remove('hidden');

  // Update all dashboard components
  updateMetrics();
  updateCharts();
  updateWordAnalysis();
  updatePostsDisplay();
  initializePostTabs(); // Update tab counts
  
  // Close sidebar on mobile after analysis
  if (window.innerWidth <= 1024) {
    document.querySelector('.sidebar').classList.remove('open');
  }
}

// Generate realistic analysis data
function generateAnalysisData() {
  const postCount = parseInt(elements.postCount.value);
  const platform = elements.platform.value;

  // Generate sentiment distribution with some randomness
  const basePositive = 42.3 + (Math.random() - 0.5) * 10;
  const baseNegative = 28.7 + (Math.random() - 0.5) * 8;
  const neutral = 100 - basePositive - baseNegative;

  currentData = {
    totalPosts: postCount,
    positive: Math.max(20, Math.min(60, basePositive)).toFixed(1),
    negative: Math.max(15, Math.min(45, baseNegative)).toFixed(1),
    neutral: Math.max(20, Math.min(50, neutral)).toFixed(1),
    confidence: (0.75 + Math.random() * 0.2).toFixed(3),
    platform: platform
  };
}

// Update metrics display
function updateMetrics() {
  const data = currentData || {
    totalPosts: 1000,
    positive: 42.3,
    negative: 28.7,
    neutral: 29.0,
    confidence: 0.834
  };

  document.getElementById('totalPosts').textContent = parseInt(data.totalPosts).toLocaleString();
  document.getElementById('positivePercent').textContent = `${data.positive}%`;
  document.getElementById('negativePercent').textContent = `${data.negative}%`;
  document.getElementById('avgConfidence').textContent = `${(parseFloat(data.confidence) * 100).toFixed(1)}%`;
}

// Initialize charts
function initializeCharts() {
  createSentimentPieChart();
  createTrendsLineChart();
  createPlatformBarChart();
}

// Create sentiment distribution pie chart
function createSentimentPieChart() {
  const ctx = document.getElementById('sentimentPieChart').getContext('2d');
  
  sentimentChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Positive', 'Negative', 'Neutral'],
      datasets: [{
        data: [42.3, 28.7, 29.0],
        backgroundColor: ['#28a745', '#dc3545', '#6c757d'],
        borderWidth: 2,
        borderColor: '#fff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            padding: 20,
            usePointStyle: true
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const label = context.label;
              const value = context.parsed;
              const total = context.dataset.data.reduce((a, b) => a + b, 0);
              const percentage = ((value / total) * 100).toFixed(1);
              return `${label}: ${percentage}% (${Math.round(value * 10)} posts)`;
            }
          }
        }
      }
    }
  });
}

// Create trends line chart
function createTrendsLineChart() {
  const ctx = document.getElementById('trendsLineChart').getContext('2d');
  
  trendsChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sampleData.trendData.map(d => new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })),
      datasets: [
        {
          label: 'Positive',
          data: sampleData.trendData.map(d => d.positive),
          borderColor: '#28a745',
          backgroundColor: 'rgba(40, 167, 69, 0.1)',
          tension: 0.4,
          fill: false
        },
        {
          label: 'Negative',
          data: sampleData.trendData.map(d => d.negative),
          borderColor: '#dc3545',
          backgroundColor: 'rgba(220, 53, 69, 0.1)',
          tension: 0.4,
          fill: false
        },
        {
          label: 'Neutral',
          data: sampleData.trendData.map(d => d.neutral),
          borderColor: '#6c757d',
          backgroundColor: 'rgba(108, 117, 125, 0.1)',
          tension: 0.4,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: 'index'
      },
      plugins: {
        legend: {
          position: 'bottom'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 50,
          ticks: {
            callback: function(value) {
              return value + '%';
            }
          }
        }
      }
    }
  });
}

// Create platform comparison bar chart
function createPlatformBarChart() {
  const ctx = document.getElementById('platformBarChart').getContext('2d');
  
  platformChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Twitter', 'Reddit'],
      datasets: [
        {
          label: 'Positive',
          data: [44.2, 40.4],
          backgroundColor: '#28a745'
        },
        {
          label: 'Negative',
          data: [27.1, 30.3],
          backgroundColor: '#dc3545'
        },
        {
          label: 'Neutral',
          data: [28.7, 29.3],
          backgroundColor: '#6c757d'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom'
        }
      },
      scales: {
        x: {
          stacked: true
        },
        y: {
          stacked: true,
          beginAtZero: true,
          max: 100,
          ticks: {
            callback: function(value) {
              return value + '%';
            }
          }
        }
      }
    }
  });
}

// Update charts with new data
function updateCharts() {
  if (sentimentChart && currentData) {
    sentimentChart.data.datasets[0].data = [
      parseFloat(currentData.positive),
      parseFloat(currentData.negative),
      parseFloat(currentData.neutral)
    ];
    sentimentChart.update('active');
  }

  // Add some variation to trends
  if (trendsChart) {
    const variation = () => Math.random() * 6 - 3;
    trendsChart.data.datasets[0].data = sampleData.trendData.map(d => Math.max(20, d.positive + variation()));
    trendsChart.data.datasets[1].data = sampleData.trendData.map(d => Math.max(15, d.negative + variation()));
    trendsChart.data.datasets[2].data = sampleData.trendData.map(d => Math.max(20, d.neutral + variation()));
    trendsChart.update('active');
  }

  // Update platform chart based on selected platform
  if (platformChart && currentData) {
    if (currentData.platform === 'twitter') {
      platformChart.data.labels = ['Twitter'];
      platformChart.data.datasets[0].data = [parseFloat(currentData.positive)];
      platformChart.data.datasets[1].data = [parseFloat(currentData.negative)];
      platformChart.data.datasets[2].data = [parseFloat(currentData.neutral)];
    } else if (currentData.platform === 'reddit') {
      platformChart.data.labels = ['Reddit'];
      platformChart.data.datasets[0].data = [parseFloat(currentData.positive)];
      platformChart.data.datasets[1].data = [parseFloat(currentData.negative)];
      platformChart.data.datasets[2].data = [parseFloat(currentData.neutral)];
    } else {
      platformChart.data.labels = ['Twitter', 'Reddit'];
      const twitterPos = parseFloat(currentData.positive) + Math.random() * 4 - 2;
      const redditPos = parseFloat(currentData.positive) + Math.random() * 4 - 2;
      platformChart.data.datasets[0].data = [twitterPos, redditPos];
      platformChart.data.datasets[1].data = [
        parseFloat(currentData.negative) + Math.random() * 3 - 1.5,
        parseFloat(currentData.negative) + Math.random() * 3 - 1.5
      ];
      platformChart.data.datasets[2].data = [
        100 - twitterPos - platformChart.data.datasets[1].data[0],
        100 - redditPos - platformChart.data.datasets[1].data[1]
      ];
    }
    platformChart.update('active');
  }
}

// Update word analysis
function updateWordAnalysis() {
  const categories = ['positive', 'negative', 'neutral'];
  
  categories.forEach(category => {
    const container = document.getElementById(`${category}Words`);
    container.innerHTML = '';
    
    sampleData.topWords[category].forEach((word, index) => {
      const tag = document.createElement('div');
      tag.className = `word-tag ${category}`;
      tag.textContent = word;
      tag.style.fontSize = `${0.9 - index * 0.02}rem`;
      container.appendChild(tag);
    });
  });
}

// Initialize post tabs
function initializePostTabs() {
  const data = currentData || { positive: 42.3, negative: 28.7, neutral: 29.0 };
  const totalPosts = parseInt(elements.postCount.value) || 1000;
  
  const counts = {
    positive: Math.round(totalPosts * parseFloat(data.positive) / 100),
    negative: Math.round(totalPosts * parseFloat(data.negative) / 100),
    neutral: Math.round(totalPosts * parseFloat(data.neutral) / 100)
  };

  document.querySelector('[data-tab="positive"]').textContent = `😊 Positive (${counts.positive.toLocaleString()})`;
  document.querySelector('[data-tab="negative"]').textContent = `😞 Negative (${counts.negative.toLocaleString()})`;
  document.querySelector('[data-tab="neutral"]').textContent = `😐 Neutral (${counts.neutral.toLocaleString()})`;
}

// Switch tabs
function switchTab(tabName) {
  // Update active tab button
  document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
  document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
  
  // Update active tab content
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
  document.getElementById(`${tabName}-posts`).classList.add('active');
}

// Update posts display
function updatePostsDisplay() {
  const sentiments = ['positive', 'negative', 'neutral'];
  
  sentiments.forEach(sentiment => {
    const container = document.getElementById(`${sentiment}-posts`);
    container.innerHTML = '';
    
    extendedPosts[sentiment].forEach(post => {
      const postElement = createPostElement(post, sentiment);
      container.appendChild(postElement);
    });
  });
}

// Create post element
function createPostElement(post, sentiment) {
  const postDiv = document.createElement('div');
  postDiv.className = `post-item ${sentiment}`;
  
  const date = new Date(post.timestamp);
  const timeAgo = getTimeAgo(date);
  
  postDiv.innerHTML = `
    <div class="post-header">
      <span class="post-platform">${post.platform}</span>
      <span class="post-confidence">Confidence: ${(post.confidence * 100).toFixed(1)}%</span>
    </div>
    <div class="post-text">${post.text}</div>
    <div class="post-meta">
      <span>${timeAgo}</span>
      <span>❤️ ${post.engagement}</span>
    </div>
  `;
  
  return postDiv;
}

// Get time ago string
function getTimeAgo(date) {
  const now = new Date();
  const diffInSeconds = Math.floor((now - date) / 1000);
  
  if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60);
    return `${minutes}m ago`;
  } else if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600);
    return `${hours}h ago`;
  } else {
    const days = Math.floor(diffInSeconds / 86400);
    return `${days}d ago`;
  }
}

// Utility functions
function simulateDownload(filename) {
  const link = document.createElement('a');
  link.href = '#';
  link.download = filename;
  link.textContent = `Downloading ${filename}...`;
  
  // Show temporary notification
  showNotification(`✅ ${filename} download started!`, 'success');
}

function copyShareLink() {
  const url = window.location.href;
  navigator.clipboard.writeText(url).then(() => {
    showNotification('🔗 Dashboard link copied to clipboard!', 'primary');
  }).catch(() => {
    showNotification('Unable to copy link to clipboard', 'error');
  });
}

function showNotification(message, type = 'success') {
  const notification = document.createElement('div');
  notification.textContent = message;
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: var(--color-${type});
    color: white;
    padding: 12px 24px;
    border-radius: 8px;
    z-index: 1000;
    animation: slideIn 0.3s ease-out;
    box-shadow: var(--shadow-md);
    max-width: 300px;
  `;
  
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.style.animation = 'slideOut 0.3s ease-out forwards';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// Add slide-in and slide-out animations
const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from {
      transform: translateX(100%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
  
  @keyframes slideOut {
    from {
      transform: translateX(0);
      opacity: 1;
    }
    to {
      transform: translateX(100%);
      opacity: 0;
    }
  }
`;
document.head.appendChild(style);