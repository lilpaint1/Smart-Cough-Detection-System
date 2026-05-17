// CoughAI — homepage.js
// Scroll reveal + nav scroll effect

document.addEventListener('DOMContentLoaded', () => {

    // ── Nav scroll effect ──────────────────────────────────
    const nav = document.getElementById('nav');
    const onScroll = () => {
        if (window.scrollY > 20) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
    };
    window.addEventListener('scroll', onScroll, { passive: true });

    // ── Scroll reveal ──────────────────────────────────────
    const revealEls = document.querySelectorAll('.scroll-reveal, .scroll-reveal-stagger');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                // Unobserve after first reveal for performance
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.15,
        rootMargin: '0px 0px -40px 0px'
    });

    revealEls.forEach(el => observer.observe(el));

    // ── Smooth scroll for anchor links ────────────────────
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', e => {
            e.preventDefault();
            const target = document.querySelector(anchor.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // ── Waveform animation randomization ─────────────────
    // Make waveform bars look more organic
    const wbars = document.querySelectorAll('.wbar');
    const randomizeWave = () => {
        wbars.forEach(bar => {
            const h = 20 + Math.random() * 75;
            bar.style.setProperty('--h', `${h}%`);
        });
    };

    // Cycle waveform heights every 1.5s for visual interest
    setInterval(randomizeWave, 1500);
});
