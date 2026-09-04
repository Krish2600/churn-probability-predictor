/* ==========================================================================
   Customer Churn Predictor - Frontend Logic & Async Interactivity
   ========================================================================== */

document.addEventListener('DOMContentLoaded', function () {
    const churnForm = document.getElementById('churnForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    
    // Result Elements
    const resultCard = document.getElementById('resultCard');
    const gaugeFill = document.getElementById('gaugeFill');
    const gaugePercentage = document.getElementById('gaugePercentage');
    const riskBadge = document.getElementById('riskBadge');
    const riskBadgeText = document.getElementById('riskBadgeText');
    const riskBadgeIcon = document.getElementById('riskBadgeIcon');
    const outcomeSummary = document.getElementById('outcomeSummary');

    // Preset Profiles Data
    const presets = {
        highRisk: {
            credit_score: 590,
            geography: 'Germany',
            gender: 'Female',
            age: 52,
            tenure: 2,
            balance: 125000,
            num_products: 1,
            has_card: '1',
            is_active: '0',
            salary: 85000
        },
        loyal: {
            credit_score: 780,
            geography: 'France',
            gender: 'Female',
            age: 34,
            tenure: 8,
            balance: 65000,
            num_products: 2,
            has_card: '1',
            is_active: '1',
            salary: 95000
        },
        moderate: {
            credit_score: 640,
            geography: 'Spain',
            gender: 'Male',
            age: 44,
            tenure: 4,
            balance: 92000,
            num_products: 3,
            has_card: '0',
            is_active: '1',
            salary: 62000
        }
    };

    // Apply Presets
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const presetKey = this.dataset.preset;
            const data = presets[presetKey];
            if (!data) return;

            document.getElementById('credit_score').value = data.credit_score;
            document.getElementById('geography').value = data.geography;
            document.getElementById('gender').value = data.gender;
            document.getElementById('age').value = data.age;
            document.getElementById('tenure').value = data.tenure;
            document.getElementById('balance').value = data.balance;
            document.getElementById('num_products').value = data.num_products;
            
            // Checkboxes
            const hasCardCb = document.getElementById('has_card_cb');
            const isActiveCb = document.getElementById('is_active_cb');
            
            if (hasCardCb) {
                hasCardCb.checked = data.has_card === '1';
                document.getElementById('has_card').value = data.has_card;
            }
            if (isActiveCb) {
                isActiveCb.checked = data.is_active === '1';
                document.getElementById('is_active').value = data.is_active;
            }

            document.getElementById('salary').value = data.salary;

            // Trigger visual glow on form
            churnForm.classList.add('animate-fade-in');
            setTimeout(() => churnForm.classList.remove('animate-fade-in'), 500);
        });
    });

    // Checkbox hidden input sync
    const hasCardCb = document.getElementById('has_card_cb');
    const isActiveCb = document.getElementById('is_active_cb');

    if (hasCardCb) {
        hasCardCb.addEventListener('change', function () {
            document.getElementById('has_card').value = this.checked ? '1' : '0';
        });
    }
    if (isActiveCb) {
        isActiveCb.addEventListener('change', function () {
            document.getElementById('is_active').value = this.checked ? '1' : '0';
        });
    }

    // Gauge Update Function
    function updateGauge(prob) {
        // Circumference of circle with r=90 is ~565.48
        const maxOffset = 565;
        const offset = maxOffset - (maxOffset * (prob / 100));
        
        let strokeColor = '#10b981'; // Green
        let badgeClass = 'risk-low';
        let badgeText = 'Low Churn Risk';
        let badgeIconClass = 'fas fa-shield-alt me-2';
        let summaryText = 'This customer exhibits strong retention signals and low likelihood of exiting.';

        if (prob >= 60) {
            strokeColor = '#ef4444'; // Red
            badgeClass = 'risk-high';
            badgeText = 'High Churn Risk';
            badgeIconClass = 'fas fa-exclamation-triangle me-2';
            summaryText = 'High probability of customer churn! Proactive retention offer recommended.';
        } else if (prob >= 35) {
            strokeColor = '#f59e0b'; // Amber
            badgeClass = 'risk-moderate';
            badgeText = 'Moderate Risk';
            badgeIconClass = 'fas fa-exclamation-circle me-2';
            summaryText = 'Customer displays moderate churn indicators. Monitor engagement closely.';
        }

        if (gaugeFill) {
            gaugeFill.style.stroke = strokeColor;
            gaugeFill.style.strokeDashoffset = offset;
        }

        if (gaugePercentage) {
            // Count up animation
            let current = 0;
            const target = prob;
            const duration = 1000;
            const stepTime = 20;
            const steps = duration / stepTime;
            const increment = target / steps;

            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                gaugePercentage.textContent = current.toFixed(1) + '%';
            }, stepTime);
        }

        if (riskBadge) {
            riskBadge.className = 'risk-badge ' + badgeClass;
            riskBadgeText.textContent = badgeText;
            riskBadgeIcon.className = badgeIconClass;
        }

        if (outcomeSummary) {
            outcomeSummary.textContent = summaryText;
        }
    }

    // Form Submission Handling (AJAX)
    if (churnForm) {
        churnForm.addEventListener('submit', function (e) {
            e.preventDefault();

            if (!churnForm.checkValidity()) {
                churnForm.classList.add('was-validated');
                return;
            }

            // UI Loading state
            submitBtn.disabled = true;
            btnText.textContent = 'Analyzing Neural Model...';
            btnSpinner.classList.remove('d-none');

            const formData = new FormData(churnForm);

            fetch('/api/predict', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                submitBtn.disabled = false;
                btnText.textContent = 'Calculate Churn Probability';
                btnSpinner.classList.add('d-none');

                if (data.success) {
                    resultCard.classList.remove('d-none');
                    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    updateGauge(data.probability);
                } else {
                    alert(data.error || 'Prediction failed. Please check your inputs.');
                }
            })
            .catch(err => {
                console.warn('AJAX request failed, submitting via standard POST fallback:', err);
                churnForm.submit();
            });
        });
    }

    // Initial check if result card was rendered via server template fallback
    const serverProb = resultCard ? resultCard.dataset.serverProb : null;
    if (serverProb && !isNaN(parseFloat(serverProb))) {
        resultCard.classList.remove('d-none');
        updateGauge(parseFloat(serverProb));
    }
});
