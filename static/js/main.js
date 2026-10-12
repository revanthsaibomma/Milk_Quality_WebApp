
"use strict";

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("milkForm");
    const analyzeButton = document.getElementById("analyzeButton");
    const colourInput = document.getElementById("colourInput");
    const colourPreview = document.getElementById("colourPreview");
    const printButton = document.getElementById("printReport");

    // Update the colour preview from the entered numeric value.
    function updateColourPreview() {
        if (!colourInput || !colourPreview) return;

        const rawValue = Number(colourInput.value);
        const value = Number.isFinite(rawValue)
            ? Math.max(0, Math.min(255, rawValue))
            : 240;

        colourPreview.style.backgroundColor =
            `rgb(${value}, ${value}, ${value})`;

        colourPreview.title = `Colour value: ${value}`;
    }

    if (colourInput) {
        colourInput.addEventListener("input", updateColourPreview);
        updateColourPreview();
    }

    // Validate the numeric ranges before submitting to Flask.
    if (form) {
        form.addEventListener("submit", (event) => {
            if (!form.checkValidity()) {
                event.preventDefault();
                form.reportValidity();
                return;
            }

            const ph = Number(form.elements["pH"].value);
            const temperature = Number(
                form.elements["Temprature"].value
            );
            const colour = Number(form.elements["Colour"].value);

            if (!Number.isFinite(ph) || ph < 0 || ph > 14) {
                event.preventDefault();
                alert("Enter a pH value between 0 and 14.");
                form.elements["pH"].focus();
                return;
            }

            if (
                !Number.isFinite(temperature) ||
                temperature < 0 ||
                temperature > 100
            ) {
                event.preventDefault();
                alert("Enter a temperature between 0°C and 100°C.");
                form.elements["Temprature"].focus();
                return;
            }

            if (
                !Number.isFinite(colour) ||
                colour < 0 ||
                colour > 255
            ) {
                event.preventDefault();
                alert("Enter a colour value between 0 and 255.");
                form.elements["Colour"].focus();
                return;
            }

            // Show a loading state while the Flask request is processing.
            if (analyzeButton && !analyzeButton.disabled) {
                analyzeButton.disabled = true;

                const label = analyzeButton.querySelector(".button-label");
                if (label) {
                    label.textContent = "Analyzing sample...";
                }

                const symbol = analyzeButton.querySelector(".button-symbol");
                if (symbol) {
                    symbol.textContent = "◌";
                }
            }
        });
    }

    // Animate probability bars when the report becomes visible.
    const progressBars = document.querySelectorAll(".progress-bar");

    progressBars.forEach((bar) => {
        const rawWidth = Number(bar.dataset.width);
        const width = Number.isFinite(rawWidth)
            ? Math.max(0, Math.min(100, rawWidth))
            : 0;

        bar.style.width = "0%";

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                bar.style.width = `${width}%`;
            });
        });
    });

    // Print the prediction report.
    if (printButton) {
        printButton.addEventListener("click", () => {
            window.print();
        });
    }

    // Highlight the navigation link corresponding to the selected section.
    const navLinks = document.querySelectorAll(".nav-link");

    navLinks.forEach((link) => {
        link.addEventListener("click", () => {
            navLinks.forEach((item) => item.classList.remove("active"));
            link.classList.add("active");
        });
    });

    // If Flask returns a prediction, scroll the report into view.
    const report = document.getElementById("results");

    if (report) {
        window.setTimeout(() => {
            report.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }, 100);
    }
});
