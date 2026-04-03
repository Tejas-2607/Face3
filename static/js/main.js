/**
 * FACE RECOGNITION SYSTEM - MAIN JAVASCRIPT
 * Common utilities and helper functions
 */

// Utility functions
const Utils = {
    /**
     * Show notification message
     */
    notify: function(message, type = 'info') {
        // You can integrate a toast library here
        console.log(`[${type.toUpperCase()}] ${message}`);
        alert(message);
    },
    
    /**
     * Format timestamp
     */
    formatTimestamp: function(date) {
        return date.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    },
    
    /**
     * Debounce function
     */
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// API helper
const API = {
    baseURL: '',
    
    /**
     * Generic fetch wrapper
     */
    request: async function(endpoint, options = {}) {
        const url = this.baseURL + endpoint;
        
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, config);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },
    
    /**
     * GET request
     */
    get: async function(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    },
    
    /**
     * POST request
     */
    post: async function(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },
    
    /**
     * DELETE request
     */
    delete: async function(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
};

// System status checker
class SystemStatus {
    constructor() {
        this.statusEndpoint = '/api/system_status';
        this.updateInterval = 5000; // 5 seconds
        this.intervalId = null;
    }
    
    async check() {
        try {
            const status = await API.get(this.statusEndpoint);
            return status;
        } catch (error) {
            console.error('Failed to check system status:', error);
            return null;
        }
    }
    
    startMonitoring(callback) {
        this.intervalId = setInterval(async () => {
            const status = await this.check();
            if (status && callback) {
                callback(status);
            }
        }, this.updateInterval);
    }
    
    stopMonitoring() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Utils, API, SystemStatus };
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Face Recognition System initialized');
    
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
});