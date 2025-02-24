// Initialize AOS
AOS.init({
    duration: 1000,
    once: true,
    offset: 100
});

// Mobile Menu Animation
document.addEventListener('DOMContentLoaded', () => {


    const navItems = document.querySelector('.nav-items');
    const navLinks = document.querySelectorAll('.nav-link');
    let isMenuOpen = false;

    function toggleMenu() {
        isMenuOpen = !isMenuOpen;
        menuBtn.classList.toggle('open');
        navItems.classList.toggle('active');
        document.body.style.overflow = isMenuOpen ? 'hidden' : '';
    }

    menuBtn.addEventListener('click', toggleMenu);

    // Close menu when clicking a link
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            if (isMenuOpen) {
                toggleMenu();
            }
        });
    });



    // Close menu on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isMenuOpen) {
            toggleMenu();
        }
    });
});

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            window.scrollTo({
                top: target.offsetTop - 70,
                behavior: 'smooth'
            });
        }
    });
});

// Navbar scroll effect
const navbar = document.querySelector('.animate-navbar');
let lastScroll = 0;

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll <= 0) {
        navbar.classList.remove('scroll-up');
        return;
    }
    
    if (currentScroll > lastScroll && !navbar.classList.contains('scroll-down')) {
        navbar.classList.remove('scroll-up');
        navbar.classList.add('scroll-down');
    } else if (currentScroll < lastScroll && navbar.classList.contains('scroll-down')) {
        navbar.classList.remove('scroll-down');
        navbar.classList.add('scroll-up');
    }
    lastScroll = currentScroll;

    if (currentScroll > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// Form validation and animation
const form = document.querySelector('.contact-form');
const inputs = form.querySelectorAll('input, textarea');

inputs.forEach(input => {
    input.addEventListener('focus', () => {
        input.parentElement.classList.add('focused');
    });
    
    input.addEventListener('blur', () => {
        if (input.value === '') {
            input.parentElement.classList.remove('focused');
        }
    });
});

// Initialize EmailJS
(function() {
    emailjs.init("YOUR_SERVICE_ID");
})();

// Handle contact form submission
document.getElementById('contactForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Get form elements
    const form = this;
    const formStatus = form.querySelector('.form-status');
    const submitBtn = form.querySelector('.submit-btn');
    const originalBtnText = submitBtn.innerHTML;
    
    // Change button state while sending
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
    
    // Prepare template parameters
    const templateParams = {
        to_name: 'Code With Panda',
        user_name: document.getElementById('name').value,
        user_email: document.getElementById('email').value,
        subject: document.getElementById('subject').value,
        message: document.getElementById('message').value
    };

    // Send email using EmailJS
    emailjs.send('YOUR_SERVICE_ID', 'YOUR_TEMPLATE_ID', templateParams)
        .then(function(response) {
            // Log success
            console.log('SUCCESS!', response.status, response.text);
            
            // Show success message
            formStatus.innerHTML = '<div class="success-message">Message sent successfully! I\'ll get back to you soon.</div>';
            
            // Reset form
            form.reset();
            
            // Reset button state after 2 seconds
            setTimeout(() => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnText;
                
                // Clear success message after 5 seconds
                setTimeout(() => {
                    formStatus.innerHTML = '';
                }, 5000);
            }, 2000);
        })
        .catch(function(error) {
            // Log error
            console.error('FAILED...', error);
            
            // Show error message
            formStatus.innerHTML = '<div class="error-message">Oops! Something went wrong. Please try again.</div>';
            
            // Reset button state
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalBtnText;
            
            // Log detailed error
            if (error.text) {
                console.error('Error details:', error.text);
            }
        });
});


// const preloader = document.querySelector('.preloader');
// const counter = document.querySelector('.counter');
// let count = 0;

// const updateCounter = () => {
//     counter.textContent = count + '%';
//     if (count < 100) {
//         count++;
//         setTimeout(updateCounter, 20);
//     } else {
//         preloader.classList.add('hidden');
//     }
// };

// updateCounter();

// Custom cursor
// const cursor = document.querySelector('.cursor');
// const cursorFollower = document.querySelector('.cursor-follower');

// document.addEventListener('mousemove', (e) => {
//     cursor.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
//     cursorFollower.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
// });

// Add hover effect to links
document.querySelectorAll('a, button').forEach(link => {
    link.addEventListener('mouseenter', () => {
        cursorFollower.style.width = '50px';
        cursorFollower.style.height = '50px';
    });
    
    link.addEventListener('mouseleave', () => {
        cursorFollower.style.width = '30px';
        cursorFollower.style.height = '30px';
    });
});


// const roles = ['Web Developer', 'UI/UX Designer', 'Full Stack Developer'];
// let roleIndex = 0;
// let charIndex = 0;
// let isDeleting = false;
// const typingDelay = 100;
// const erasingDelay = 50;
// const newRoleDelay = 2000;

// function typeRole() {
//     const currentRole = roles[roleIndex];
//     const typedText = document.querySelector('.typed-text');
    
//     if (isDeleting) {
//         charIndex--;
//         typedText.textContent = currentRole.substring(0, charIndex);
//     } else {
//         charIndex++;
//         typedText.textContent = currentRole.substring(0, charIndex);
//     }
    
//     let typeSpeed = isDeleting ? erasingDelay : typingDelay;
    
//     if (!isDeleting && charIndex === currentRole.length) {
//         typeSpeed = newRoleDelay;
//         isDeleting = true;
//     } else if (isDeleting && charIndex === 0) {
//         isDeleting = false;
//         roleIndex = (roleIndex + 1) % roles.length;
//     }
    
//     setTimeout(typeRole, typeSpeed);
// }

// typeRole();

// Skill bars animation
const skillBars = document.querySelectorAll('.skill-progress');

const animateSkillBars = () => {
    skillBars.forEach(bar => {
        const progress = bar.getAttribute('data-progress');
        bar.style.transform = `scaleX(${progress / 100})`;
    });
};

// Animate skill bars when they come into view
const skillsSection = document.querySelector('.skills');
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            animateSkillBars();
        }
    });
});

observer.observe(skillsSection);

// Noise effect
const noise = document.getElementById('noise');
const ctx = noise.getContext('2d');

noise.width = window.innerWidth;
noise.height = window.innerHeight;

function generateNoise() {
    const imageData = ctx.createImageData(noise.width, noise.height);
    const data = imageData.data;
    
    for (let i = 0; i < data.length; i += 4) {
        const value = Math.random() * 255;
        data[i] = value;
        data[i + 1] = value;
        data[i + 2] = value;
        data[i + 3] = 15;
    }
    
    ctx.putImageData(imageData, 0, 0);
}

function animateNoise() {
    generateNoise();
    requestAnimationFrame(animateNoise);
}

animateNoise();

// Project Modal


const modalOverlay = document.querySelector('.modal-overlay');
const modal = document.querySelector('.modal');
const modalClose = document.querySelector('.modal-close');
const projectCards = document.querySelectorAll('.project-card');

function openModal(project) {
    const modalImage = modal.querySelector('.modal-image');
    const modalThumbnails = modal.querySelector('.modal-thumbnails');
    const modalTitle = modal.querySelector('.modal-title');
    const modalDescription = modal.querySelector('.modal-description');
    const modalTags = modal.querySelector('.modal-tags');
    const modalLinks = modal.querySelector('.modal-links');
    
    // Set main image and thumbnails
    modalImage.src = project.images[0];
    modalThumbnails.innerHTML = project.images.map((img, index) => `
        <img src="${img}" alt="Thumbnail ${index + 1}" 
             class="thumbnail ${index === 0 ? 'active' : ''}"
             onclick="changeImage(${index}, this)">
    `).join('');
    
    modalTitle.textContent = project.title;
    modalDescription.textContent = project.description;
    
    // Clear and add new tags
    modalTags.innerHTML = project.tags.map(tag => `
        <span class="modal-tag">${tag}</span>
    `).join('');
    
    // Update links
    const [liveLink, codeLink] = modalLinks.querySelectorAll('a');
    liveLink.href = project.liveLink;
    codeLink.href = project.codeLink;
    
    modalOverlay.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function changeImage(index, thumbnail) {
    const modalImage = modal.querySelector('.modal-image');
    const thumbnails = modal.querySelectorAll('.thumbnail');
    
    modalImage.src = thumbnail.src;
    thumbnails.forEach(thumb => thumb.classList.remove('active'));
    thumbnail.classList.add('active');
}

function closeModal() {
    modalOverlay.classList.remove('active');
    document.body.style.overflow = '';
}

projectCards.forEach((card, index) => {
    card.addEventListener('click', () => openModal(projects[index]));
});

modalClose.addEventListener('click', closeModal);
modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) closeModal();
});

// Enhanced scroll animations
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const sections = document.querySelectorAll('section');
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop - 300;
        if (scrolled >= sectionTop) {
            section.style.opacity = '1';
            section.style.transform = 'translateY(0)';
        }
    });
});

// Form animations
const formInputs = document.querySelectorAll('.input-container input, .input-container textarea');

formInputs.forEach(input => {
    input.addEventListener('focus', () => {
        input.parentElement.classList.add('focused');
    });
    
    input.addEventListener('blur', () => {
        if (!input.value) {
            input.parentElement.classList.remove('focused');
        }
    });
});

// Resume button click handler
document.getElementById('resumeBtn').addEventListener('click', function(e) {
    e.preventDefault();
    window.location.href = 'resume.html';
});

// Initialize particles
function initParticles() {
    const particles = document.querySelectorAll('.particle');
    particles.forEach((particle, index) => {
        const delay = Math.random() * 5;
        const duration = 15 + Math.random() * 10;
        const size = 3 + Math.random() * 5;
        
        particle.style.left = Math.random() * 100 + 'vw';
        particle.style.animationDelay = delay + 's';
        particle.style.animationDuration = duration + 's';
        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
    });
}

// Navbar scroll effect
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.animate-navbar');
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// Initialize particles on load
document.addEventListener('DOMContentLoaded', initParticles);

document.addEventListener('DOMContentLoaded', () => {
    const skillBoxes = document.querySelectorAll('.skill-box');
    
    skillBoxes.forEach(box => {
        const progressBar = box.querySelector('.progress-bar');
        const progress = progressBar.getAttribute('data-progress');
        progressBar.style.setProperty('--progress', progress);
    });

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
            }
        });
    }, { threshold: 0.5 });

    skillBoxes.forEach(box => observer.observe(box));
});

// Stack Modal Functionality
const stackCount = document.querySelector('.stack-count');
const stackModal = document.querySelector('.stack-modal');
const closeModal = document.querySelector('.close-modal');

stackCount.addEventListener('click', () => {
    stackModal.classList.add('active');
    document.body.style.overflow = 'hidden';
});

closeModal.addEventListener('click', () => {
    stackModal.classList.remove('active');
    document.body.style.overflow = 'auto';
});

stackModal.addEventListener('click', (e) => {
    if (e.target === stackModal) {
        stackModal.classList.remove('active');
        document.body.style.overflow = 'auto';
    }
});

// Close modal on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && stackModal.classList.contains('active')) {
        stackModal.classList.remove('active');
        document.body.style.overflow = 'auto';
    }
});






