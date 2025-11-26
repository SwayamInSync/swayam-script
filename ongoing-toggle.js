// Ongoing Posts Toggle Functionality
(function() {
  'use strict';
  
  // Initialize the toggle state from localStorage
  function getOngoingToggleState() {
    const stored = localStorage.getItem('show-ongoing-posts');
    return stored === 'true'; // Default is false (OFF)
  }
  
  function setOngoingToggleState(state) {
    localStorage.setItem('show-ongoing-posts', state.toString());
  }
  
  // Filter posts based on ongoing toggle state
  function filterPosts() {
    const showOngoing = getOngoingToggleState();
    const posts = document.querySelectorAll('.quarto-post');
    
    posts.forEach(post => {
      const categories = post.getAttribute('data-categories') || '';
      // Decode base64 categories and check for "Ongoing"
      const decodedCategories = categories.split(',').map(cat => {
        try {
          return atob(cat);
        } catch(e) {
          return '';
        }
      }).join(',').toLowerCase();
      
      const isOngoing = decodedCategories.includes('ongoing');
      
      if (!showOngoing && isOngoing) {
        post.style.display = 'none';
      } else {
        post.style.display = '';
      }
    });
    
    // Update category counts
    updateCategoryCounts();
  }
  
  // Update category counts after filtering
  function updateCategoryCounts() {
    const showOngoing = getOngoingToggleState();
    const categoryElements = document.querySelectorAll('.quarto-listing-category .category');
    
    categoryElements.forEach(categoryEl => {
      const categoryValue = categoryEl.getAttribute('data-category');
      
      if (!categoryValue) {
        // "All" category - count visible posts
        const visiblePosts = document.querySelectorAll('.quarto-post[style=""], .quarto-post:not([style*="display"])');
        const countEl = categoryEl.querySelector('.quarto-category-count');
        if (countEl) {
          countEl.textContent = `(${visiblePosts.length})`;
        }
      } else {
        // Specific category - count visible posts with that category
        const posts = document.querySelectorAll(`.quarto-post[data-categories*="${categoryValue}"]`);
        let visibleCount = 0;
        posts.forEach(post => {
          if (post.style.display !== 'none') {
            visibleCount++;
          }
        });
        const countEl = categoryEl.querySelector('.quarto-category-count');
        if (countEl) {
          countEl.textContent = `(${visibleCount})`;
        }
      }
    });
  }
  
  // Toggle the ongoing posts visibility
  function toggleOngoingPosts() {
    const currentState = getOngoingToggleState();
    const newState = !currentState;
    setOngoingToggleState(newState);
    updateToggleButton();
    filterPosts();
  }
  
  // Update the toggle button appearance
  function updateToggleButton() {
    const button = document.querySelector('.ongoing-toggle');
    if (!button) return;
    
    const isOn = getOngoingToggleState();
    const icon = button.querySelector('i');
    
    if (isOn) {
      button.classList.add('active');
      icon.classList.remove('bi-eye-slash');
      icon.classList.add('bi-eye');
      button.title = 'Hide ongoing posts';
    } else {
      button.classList.remove('active');
      icon.classList.remove('bi-eye');
      icon.classList.add('bi-eye-slash');
      button.title = 'Show ongoing posts';
    }
  }
  
  // Create and insert the toggle button
  function createToggleButton() {
    const navbarTools = document.querySelector('.quarto-navbar-tools');
    if (!navbarTools) return;
    
    // Check if button already exists
    if (document.querySelector('.ongoing-toggle')) return;
    
    const button = document.createElement('a');
    button.href = '';
    button.className = 'ongoing-toggle quarto-navigation-tool px-1';
    button.onclick = function(e) {
      e.preventDefault();
      toggleOngoingPosts();
      return false;
    };
    
    const icon = document.createElement('i');
    icon.className = 'bi';
    button.appendChild(icon);
    
    // Insert before the color scheme toggle
    const colorToggle = navbarTools.querySelector('.quarto-color-scheme-toggle');
    if (colorToggle) {
      navbarTools.insertBefore(button, colorToggle);
    } else {
      navbarTools.appendChild(button);
    }
    
    updateToggleButton();
  }
  
  // Initialize when DOM is ready
  function init() {
    createToggleButton();
    filterPosts();
    
    // Re-filter when quarto listing is loaded
    if (window['quarto-listing-loaded']) {
      const originalLoaded = window['quarto-listing-loaded'];
      window['quarto-listing-loaded'] = function() {
        originalLoaded();
        filterPosts();
      };
    } else {
      window['quarto-listing-loaded'] = function() {
        filterPosts();
      };
    }
  }
  
  // Wait for DOM to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
  // Re-initialize after category filter changes
  document.addEventListener('click', function(e) {
    if (e.target.classList.contains('category') || 
        e.target.classList.contains('listing-category')) {
      setTimeout(filterPosts, 100);
    }
  });
})();
