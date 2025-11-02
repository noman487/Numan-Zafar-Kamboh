import type { AspectRatio } from '../types';

const STABILITY_API_KEY = process.env.STABILITY_API_KEY;
const API_HOST = 'https://api.stability.ai';
const ENGINE_ID = 'stable-image-generate-sd3';

export async function generateImageWithStability(prompt: string, aspectRatio: AspectRatio, negativePrompt: string | undefined, numImages: number): Promise<{ base64: string | null; error: string | null; }[]> {
    if (!STABILITY_API_KEY) {
        return Array(numImages).fill({ base64: null, error: "Stability AI API key is not configured." });
    }

    const formData = new FormData();
    formData.append('prompt', prompt);
    formData.append('aspect_ratio', aspectRatio);
    formData.append('output_format', 'jpeg');
    if (numImages > 1) {
        formData.append('samples', String(numImages));
    }
    if (negativePrompt && negativePrompt.trim()) {
        formData.append('negative_prompt', negativePrompt.trim());
    }

    try {
        const response = await fetch(
            `${API_HOST}/v2beta/stable-image/generate/${ENGINE_ID}`,
            {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${STABILITY_API_KEY}`,
                    Accept: 'application/json',
                },
                body: formData,
            }
        );

        if (!response.ok) {
            const errorBody = await response.json();
            const errorMessage = errorBody.errors ? errorBody.errors[0] : `Unknown Stability AI error (${response.status})`;
            console.error('Stability AI Error:', errorMessage);
            throw new Error(errorMessage);
        }

        const jsonResponse = await response.json();
        
        if (jsonResponse.images && jsonResponse.images.length > 0) {
            const results = jsonResponse.images.map((image: { base64: string }) => ({ base64: image.base64, error: null }));
            while (results.length < numImages) {
                results.push({ base64: null, error: 'API returned fewer images than requested.' });
            }
            return results;
        } else {
            return Array(numImages).fill({ base64: null, error: 'Stability AI did not return any images.' });
        }

    } catch (err) {
        console.error('Stability AI request failed:', err);
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred with Stability AI.';
        return Array(numImages).fill({ base64: null, error: `Stability AI Error: ${errorMessage}` });
    }
}
