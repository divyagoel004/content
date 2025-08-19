document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generate-btn');
    const topicInput = document.getElementById('topic-input');
    const loadingSpinner = document.getElementById('loading-spinner');
    const slideDisplay = document.getElementById('slide-display');
    const presentationContainer = document.getElementById('presentation-container');
    const navigationControls = document.getElementById('navigation-controls');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const slideCounter = document.getElementById('slide-counter');

    let presentationData = null;
    let currentSlideIndex = 0;

    const API_BASE_URL = 'http://localhost:8000'; // Assuming backend runs on port 8000

    generateBtn.addEventListener('click', async () => {
        const topic = topicInput.value;
        if (!topic.trim()) {
            alert('Please enter a topic.');
            return;
        }

        loadingSpinner.style.display = 'block';
        slideDisplay.style.display = 'none';
        navigationControls.style.display = 'none';

        try {
            const response = await fetch(`${API_BASE_URL}/api/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ topic: topic, depth_level: 'intermediate' }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to generate presentation.');
            }

            presentationData = await response.json();

            if (presentationData && presentationData.slides && presentationData.slides.length > 0) {
                currentSlideIndex = 0;
                renderSlide(currentSlideIndex);
                navigationControls.style.display = 'block';
                slideDisplay.style.display = 'block';
            } else {
                alert('The generated presentation is empty.');
            }

        } catch (error) {
            alert(`An error occurred: ${error.message}`);
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });

    const renderSlide = (index) => {
        const slide = presentationData.slides[index];
        let contentHtml = `
            <div class="slide-header">
                ${slide.title}
                <div class="slide-number">${slide.slide_number}/${presentationData.slides.length}</div>
            </div>
            <div class="slide-content">
        `;

        slide.content_sections.forEach(section => {
            switch (section.type) {
                case 'text':
                    contentHtml += `<p>${section.content.replace(/\n/g, '<br>')}</p>`;
                    break;
                case 'code':
                    contentHtml += `<pre><code>${section.content}</code></pre>`;
                    break;
                case 'image':
                    // Image path is relative to the static dir of the backend
                    contentHtml += `<img src="${API_BASE_URL}/static/${section.content}" alt="${section.component_name}">`;
                    break;
                default:
                    contentHtml += `<p>${section.content}</p>`;
            }
        });

        contentHtml += '</div>';
        slideDisplay.innerHTML = contentHtml;
        updateNavigation();
    };

    const updateNavigation = () => {
        slideCounter.textContent = `${currentSlideIndex + 1} / ${presentationData.slides.length}`;
        prevBtn.disabled = currentSlideIndex === 0;
        nextBtn.disabled = currentSlideIndex === presentationData.slides.length - 1;
    };

    prevBtn.addEventListener('click', () => {
        if (currentSlideIndex > 0) {
            currentSlideIndex--;
            renderSlide(currentSlideIndex);
        }
    });

    nextBtn.addEventListener('click', () => {
        if (currentSlideIndex < presentationData.slides.length - 1) {
            currentSlideIndex++;
            renderSlide(currentSlideIndex);
        }
    });
});
