import os
import time

from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from django.views import View
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.utils.decorators import method_decorator

import numpy as np
from PIL import Image

from .models import ConversionJob
from .forms import ImageUploadForm
from .services import ConversionService, _get_pipeline


class UploadView(View):
    def get(self, request):
        form = ImageUploadForm()
        recent_jobs = ConversionJob.objects.filter(
            status=ConversionJob.Status.COMPLETED
        )[:5]
        return render(request, 'converter/upload.html', {
            'form': form,
            'recent_jobs': recent_jobs,
        })

    def post(self, request):
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            languages = form.cleaned_data.get('languages') or ['fa', 'en']
            job = ConversionJob.objects.create(
                original_image=form.cleaned_data['image'],
                languages=languages,
            )
            service = ConversionService()
            service.process_job(job)
            return redirect('converter:result', job_id=job.id)

        return render(request, 'converter/upload.html', {'form': form})


class ResultView(View):
    def get(self, request, job_id):
        job = get_object_or_404(ConversionJob, id=job_id)
        response = render(request, 'converter/result.html', {'job': job})
        # Allow inline scripts and iframe content
        response['Content-Security-Policy'] = "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:;"
        return response


class PreviewView(View):
    def get(self, request, job_id):
        job = get_object_or_404(ConversionJob, id=job_id)
        response = HttpResponse(job.combined_output, content_type='text/html; charset=utf-8')
        # Allow this page to be loaded in iframe and run scripts
        response['X-Frame-Options'] = 'SAMEORIGIN'
        response['Content-Security-Policy'] = "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:;"
        return response


class DebugPageView(View):
    """Three-panel debug page: original | vision overlay | generated output."""

    def get(self, request, job_id):
        job = get_object_or_404(ConversionJob, id=job_id)
        return render(request, 'converter/debug.html', {'job': job})


class DebugOverlayView(View):
    """
    Returns a PNG with detected regions/text drawn on the original image.
    Color coding:
      Blue  = depth-0 regions | Green = depth-1 | Yellow = depth-2
      Magenta = text elements (OCR)
    """

    _DEPTH_COLORS = [
        (255, 80, 80),    # depth 0 — blue
        (80, 200, 80),    # depth 1 — green
        (200, 200, 50),   # depth 2 — yellow
        (200, 80, 200),   # depth 3 — purple
    ]
    _TEXT_COLOR = (255, 50, 255)

    def get(self, request, job_id):
        import cv2
        job = get_object_or_404(ConversionJob, id=job_id)

        image = Image.open(job.original_image.path).convert('RGB')
        img = np.array(image)
        debug = img.copy()

        pipeline = _get_pipeline()
        context = pipeline.run(img)

        self._draw_regions(debug, context.regions, depth=0)

        for te in context.text_elements:
            b = te.bbox
            text = getattr(te, 'text', '') or ''
            label = text[:20] + ('…' if len(text) > 20 else '')
            cv2.rectangle(debug, (b.x, b.y), (b.x2, b.y2),
                          self._TEXT_COLOR, 1)
            if label:
                cv2.putText(debug, label, (b.x, max(b.y - 3, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            self._TEXT_COLOR, 1, cv2.LINE_AA)

        bgr = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode('.png', bgr)
        if not ok:
            return HttpResponse('Failed to encode image', status=500)
        return HttpResponse(bytes(buf), content_type='image/png')

    def _draw_regions(self, img, elements, depth):
        import cv2
        color = self._DEPTH_COLORS[min(depth, len(self._DEPTH_COLORS) - 1)]
        thickness = max(3 - depth, 1)
        for elem in elements:
            b = elem.bbox
            cv2.rectangle(img, (b.x, b.y), (b.x2, b.y2), color, thickness)
            label = f'd{depth}'
            if elem.color and elem.color.background:
                r, g, bl = elem.color.background
                label += f' #{r:02x}{g:02x}{bl:02x}'
            cv2.putText(img, label, (b.x + 2, b.y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            self._draw_regions(img, elem.children, depth + 1)


class DebugDataView(View):
    """
    Returns a JSON snapshot of what the vision pipeline detected.
    Compare this data with the original image to find where errors come from.
    """

    def get(self, request, job_id):
        job = get_object_or_404(ConversionJob, id=job_id)

        image = Image.open(job.original_image.path).convert('RGB')
        img = np.array(image)

        pipeline = _get_pipeline()
        context = pipeline.run(img)

        def bbox_dict(b):
            return {'x': b.x, 'y': b.y, 'x2': b.x2, 'y2': b.y2,
                    'w': b.width, 'h': b.height}

        def color_dict(c):
            if not c:
                return None
            return {
                'bg': list(c.background) if c.background else None,
                'fg': list(c.foreground) if c.foreground else None,
                'border': list(c.border) if c.border else None,
            }

        def elem_dict(e, depth=0):
            return {
                'id': e.id,
                'type': str(e.element_type),
                'depth': depth,
                'bbox': bbox_dict(e.bbox),
                'color': color_dict(e.color),
                'children': [elem_dict(c, depth + 1) for c in e.children],
            }

        orphan_texts = [t for t in context.text_elements if t.parent_id is None]

        data = {
            'image_size': {'w': context.width, 'h': context.height},
            'background': list(context.page_background) if context.page_background else None,
            'regions_count': len(context.regions),
            'text_elements_count': len(context.text_elements),
            'orphan_texts_count': len(orphan_texts),
            'regions': [elem_dict(r) for r in context.regions],
            'text_elements': [
                {
                    'id': t.id,
                    'text': getattr(t, 'text', ''),
                    'bbox': bbox_dict(t.bbox),
                    'color': color_dict(t.color),
                    'parent_id': t.parent_id,
                    'direction': str(getattr(t, 'direction', '')),
                }
                for t in context.text_elements
            ],
        }
        return JsonResponse(data, json_dumps_params={'indent': 2, 'ensure_ascii': False})


class DebugCompareDataView(View):
    """
    Returns a JSON analysis comparing pipeline output quality.
    Scores structure, layout, color, and semantic quality.
    """

    def get(self, request, job_id):
        import cv2
        job = get_object_or_404(ConversionJob, id=job_id)

        image = Image.open(job.original_image.path).convert('RGB')
        img = np.array(image)
        h, w = img.shape[:2]

        pipeline = _get_pipeline()
        context = pipeline.run(img)

        from vision_engine.models.page import PageStructure
        from vision_engine.models.elements import ElementType
        from vision_engine.models.layout import LayoutMode

        orphan_texts = [t for t in context.text_elements if t.parent_id is None]
        all_root = context.regions + orphan_texts
        root_elements = sorted(all_root, key=lambda e: e.bbox.area, reverse=True)

        page = PageStructure(
            width=context.width,
            height=context.height,
            background_color=context.page_background,
            root_elements=root_elements,
            text_direction=getattr(context, 'page_direction', 'ltr'),
        )

        all_elements = list(page.walk())

        # --- Element summary ---
        type_counts = {}
        for elem in all_elements:
            t = elem.element_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        element_summary = {t: {'count': c} for t, c in type_counts.items()}

        # --- Layout analysis ---
        layout_analysis = []
        layout_modes = {}
        for elem in all_elements:
            layout = getattr(elem, 'layout', None)
            if layout:
                mode = layout.mode.value if hasattr(layout.mode, 'value') else str(layout.mode)
                layout_modes[mode] = layout_modes.get(mode, 0) + 1

        has_flex = 'flex_row' in layout_modes or 'flex_column' in layout_modes
        has_grid = 'grid' in layout_modes

        if has_flex:
            flex_count = layout_modes.get('flex_row', 0) + layout_modes.get('flex_column', 0)
            layout_analysis.append({
                'level': 'good',
                'message': f'Flexbox detected: {flex_count} containers use flex layout'
            })
        else:
            layout_analysis.append({
                'level': 'warn',
                'message': 'No flexbox detected — all elements use absolute positioning'
            })

        if has_grid:
            layout_analysis.append({
                'level': 'good',
                'message': f'CSS Grid detected: {layout_modes["grid"]} containers'
            })

        block_count = layout_modes.get('block', 0)
        total_with_layout = sum(layout_modes.values())
        if block_count > 0 and total_with_layout > 0:
            ratio = block_count / total_with_layout
            if ratio > 0.8:
                layout_analysis.append({
                    'level': 'warn',
                    'message': f'{block_count}/{total_with_layout} containers use block (absolute) layout'
                })
            else:
                layout_analysis.append({
                    'level': 'good',
                    'message': f'Only {block_count}/{total_with_layout} containers fall back to absolute'
                })

        # --- Issues detection ---
        issues = []

        if len(orphan_texts) > 0:
            severity = 'warning' if len(orphan_texts) <= 3 else 'error'
            issues.append({
                'category': 'Hierarchy',
                'severity': severity,
                'message': f'{len(orphan_texts)} text elements are orphans (not in any container)'
            })

        # Check element coverage
        total_pixel_area = w * h
        covered = sum(e.bbox.area for e in context.regions)
        coverage_pct = min(100, int(covered / total_pixel_area * 100))

        if coverage_pct < 30:
            issues.append({
                'category': 'Coverage',
                'severity': 'error',
                'message': f'Only {coverage_pct}% of image area is covered by detected regions'
            })
        elif coverage_pct < 60:
            issues.append({
                'category': 'Coverage',
                'severity': 'warning',
                'message': f'{coverage_pct}% of image area is covered by detected regions'
            })

        button_count = type_counts.get('button', 0)
        input_count = type_counts.get('input', 0)
        image_count = type_counts.get('image', 0)

        if button_count == 0 and input_count == 0 and image_count == 0:
            issues.append({
                'category': 'Classification',
                'severity': 'info',
                'message': 'No interactive elements (buttons/inputs/images) detected'
            })

        # Overlapping siblings
        overlap_count = 0
        for elem in all_elements:
            if len(elem.children) < 2:
                continue
            for i in range(len(elem.children)):
                for j in range(i + 1, len(elem.children)):
                    ratio = elem.children[i].bbox.overlap_ratio(elem.children[j].bbox)
                    if ratio > 0.3:
                        overlap_count += 1
        if overlap_count > 0:
            issues.append({
                'category': 'Layout',
                'severity': 'warning',
                'message': f'{overlap_count} sibling element pairs have significant overlap (>30% IoU)'
            })

        # --- Semantic tags ---
        tag_counts = {}
        for elem in all_elements:
            tag = getattr(elem, 'semantic_tag', 'div') or 'div'
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # --- CSS features ---
        css_features = [
            {
                'name': 'Flexbox',
                'used': has_flex,
                'detail': f'({layout_modes.get("flex_row", 0)} row, {layout_modes.get("flex_column", 0)} column)' if has_flex else ''
            },
            {
                'name': 'Grid',
                'used': has_grid,
                'detail': f'({layout_modes.get("grid", 0)} containers)' if has_grid else ''
            },
            {
                'name': 'Border Radius',
                'used': any(getattr(e, 'border_radius', None) for e in all_elements),
                'detail': ''
            },
            {
                'name': 'Image Embedding',
                'used': any(
                    getattr(e, 'image_data_uri', None) for e in all_elements
                    if e.element_type == ElementType.IMAGE
                ),
                'detail': f'({sum(1 for e in all_elements if e.element_type == ElementType.IMAGE and getattr(e, "image_data_uri", None))} images)'
            },
        ]

        # --- Quality score ---
        score_parts = {}

        # Structure score (25 pts)
        structure = 25
        if coverage_pct < 30:
            structure -= 15
        elif coverage_pct < 60:
            structure -= 8
        if len(orphan_texts) > 5:
            structure -= 5
        elif len(orphan_texts) > 0:
            structure -= 2
        score_parts['structure'] = max(0, structure)

        # Layout score (25 pts)
        layout_score = 25
        if not has_flex and not has_grid:
            layout_score -= 15
        if overlap_count > 3:
            layout_score -= 5
        elif overlap_count > 0:
            layout_score -= 2
        score_parts['layout'] = max(0, layout_score)

        # Classification score (25 pts)
        classification = 25
        unique_types = len(type_counts)
        if unique_types <= 2:
            classification -= 10
        elif unique_types <= 3:
            classification -= 5
        unique_tags = len([t for t in tag_counts if t != 'div'])
        if unique_tags == 0:
            classification -= 10
        elif unique_tags <= 2:
            classification -= 3
        score_parts['classification'] = max(0, classification)

        # Visual score (25 pts)
        visual = 25
        bg = context.page_background
        if bg == (255, 255, 255) or bg == (0, 0, 0):
            visual -= 2
        elements_with_color = sum(1 for e in all_elements if e.color)
        if elements_with_color < len(all_elements) * 0.5:
            visual -= 10
        score_parts['visual'] = max(0, visual)

        total_score = sum(score_parts.values())

        data = {
            'job_id': str(job.id),
            'image_size': {'w': w, 'h': h},
            'quality_score': total_score,
            'score_breakdown': score_parts,
            'element_summary': element_summary,
            'layout_analysis': layout_analysis,
            'issues': issues,
            'semantic_tags': tag_counts,
            'css_features': css_features,
            'coverage_percent': coverage_pct,
            'total_elements': len(all_elements),
            'processing_time_ms': job.processing_time_ms,
        }

        return JsonResponse(data, json_dumps_params={'indent': 2, 'ensure_ascii': False})


class APIConvertView(View):
    def post(self, request):
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image provided'}, status=400)

        languages = request.POST.getlist('languages') or ['fa', 'en']
        job = ConversionJob.objects.create(
            original_image=request.FILES['image'],
            languages=languages,
        )

        service = ConversionService()
        service.process_job(job)

        return JsonResponse({
            'job_id': str(job.id),
            'status': job.status,
            'html': job.generated_html,
            'css': job.generated_css,
            'processing_time_ms': job.processing_time_ms,
        })


class APITestFromFolder(View):
    """
    Process all images in test_samples/ folder via API.
    GET /api/test/ — returns JSON with results for each image.
    """

    def get(self, request):
        import glob as glob_mod
        from vision_engine.models.elements import ElementType
        from vision_engine.models.page import PageStructure
        from code_generator.generator import CodeGenerator

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        samples_dir = os.path.join(project_root, 'test_samples')

        if not os.path.isdir(samples_dir):
            return JsonResponse({
                'error': f'test_samples/ folder not found',
                'hint': 'Create test_samples/ in project root and add images'
            }, status=404)

        extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob_mod.glob(os.path.join(samples_dir, ext)))

        if not image_paths:
            return JsonResponse({
                'error': 'No images found in test_samples/',
                'path': samples_dir,
            }, status=404)

        pipeline = _get_pipeline()
        generator = CodeGenerator()

        results = []
        output_dir = os.path.join(samples_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        for path in sorted(image_paths):
            try:
                img = Image.open(path).convert('RGB')
                img_array = np.array(img)
                start = time.monotonic()

                context = pipeline.run(img_array)

                orphan_texts = [t for t in context.text_elements if t.parent_id is None]
                all_root = context.regions + orphan_texts
                root_elements = sorted(all_root, key=lambda e: e.bbox.area, reverse=True)

                page = PageStructure(
                    width=context.width,
                    height=context.height,
                    background_color=context.page_background,
                    root_elements=root_elements,
                    text_direction=getattr(context, 'page_direction', 'ltr'),
                )

                result = generator.generate(page)
                elapsed_ms = int((time.monotonic() - start) * 1000)

                type_counts = {}
                for elem in page.walk():
                    t = elem.element_type.value
                    type_counts[t] = type_counts.get(t, 0) + 1

                base_name = os.path.splitext(os.path.basename(path))[0]
                html_path = os.path.join(output_dir, f'{base_name}_output.html')
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(result['combined'])

                results.append({
                    'file': os.path.basename(path),
                    'status': 'ok',
                    'image_size': {
                        'w': img_array.shape[1],
                        'h': img_array.shape[0]
                    },
                    'processing_time_ms': elapsed_ms,
                    'total_elements': sum(type_counts.values()),
                    'element_types': type_counts,
                    'regions': len(context.regions),
                    'text_elements': len(context.text_elements),
                    'output_html': html_path,
                })

            except Exception as e:
                results.append({
                    'file': os.path.basename(path),
                    'status': 'error',
                    'error': str(e),
                })

        return JsonResponse({
            'total_images': len(image_paths),
            'successful': sum(1 for r in results if r['status'] == 'ok'),
            'results': results,
        }, json_dumps_params={'indent': 2, 'ensure_ascii': False})
