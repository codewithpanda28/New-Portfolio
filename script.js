// Initialize AOS
AOS.init({
    duration: 1000,
    once: true,
    offset: 100
});

// Mobile Menu Toggle
const menuBtn = document.querySelector('.menu-btn');
const navItems = document.querySelector('.nav-items');

menuBtn.addEventListener('click', () => {
    menuBtn.classList.toggle('active');
    navItems.classList.toggle('active');
});

// Close menu when clicking a link
document.querySelectorAll('.nav-items a').forEach(link => {
    link.addEventListener('click', () => {
        menuBtn.classList.remove('active');
        navItems.classList.remove('active');
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
const navbar = document.querySelector('.navbar');
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

// Preloader
const preloader = document.querySelector('.preloader');
const counter = document.querySelector('.counter');
let count = 0;

const updateCounter = () => {
    counter.textContent = count + '%';
    if (count < 100) {
        count++;
        setTimeout(updateCounter, 20);
    } else {
        preloader.classList.add('hidden');
    }
};

updateCounter();

// Custom cursor
const cursor = document.querySelector('.cursor');
const cursorFollower = document.querySelector('.cursor-follower');

document.addEventListener('mousemove', (e) => {
    cursor.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
    cursorFollower.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
});

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

// Typing animation
const roles = ['Web Developer', 'UI/UX Designer', 'Full Stack Developer'];
let roleIndex = 0;
let charIndex = 0;
let isDeleting = false;
const typingDelay = 100;
const erasingDelay = 50;
const newRoleDelay = 2000;

function typeRole() {
    const currentRole = roles[roleIndex];
    const typedText = document.querySelector('.typed-text');
    
    if (isDeleting) {
        charIndex--;
        typedText.textContent = currentRole.substring(0, charIndex);
    } else {
        charIndex++;
        typedText.textContent = currentRole.substring(0, charIndex);
    }
    
    let typeSpeed = isDeleting ? erasingDelay : typingDelay;
    
    if (!isDeleting && charIndex === currentRole.length) {
        typeSpeed = newRoleDelay;
        isDeleting = true;
    } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        roleIndex = (roleIndex + 1) % roles.length;
    }
    
    setTimeout(typeRole, typeSpeed);
}

typeRole();

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
const projects = [
    {
        id: 1,
        title: 'E-Commerce Platform',
        description: 'A full-stack e-commerce solution built with React and Node.js. Features include user authentication, product management, shopping cart, payment integration, and order tracking. The platform is fully responsive and provides a seamless shopping experience.',
        images: [
            'https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=800',
            'https://images.unsplash.com/photo-1472851294608-062f824d29cc?auto=format&fit=crop&w=800',
            'https://images.unsplash.com/photo-1554774853-719586f82d77?auto=format&fit=crop&w=800'
        ],
        tags: ['React', 'Node.js', 'MongoDB', 'Express', 'Redux', 'Stripe'],
        liveLink: '#',
        codeLink: '#'
    },
    {
        id: 2,
        title: 'Task Management App',
        description: 'A beautiful and intuitive task management application that helps users organize their daily activities. Built with React and Firebase, it includes features like task categories, due dates, priority levels, and real-time updates.',
        images: [
            'https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?auto=format&fit=crop&w=800',
            'https://images.unsplash.com/photo-1484480974693-6ca0a78fb36b?auto=format&fit=crop&w=800',
            'https://images.unsplash.com/photo-1507238691740-187a5b1d37b8?auto=format&fit=crop&w=800'
        ],
        tags: ['React', 'Firebase', 'Material-UI', 'Context API', 'Cloud Functions'],
        liveLink: '#',
        codeLink: '#'
    }
];

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
