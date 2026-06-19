from pathlib import Path

import cv2


THRESHOLD = 0.30
MARGIN = 0.10


def create_edge_image(image):
    """
    画像から輪郭だけを取り出す。
    明るさや色の影響を減らすために使用する。
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    return edges


def crop_result_area(image):
    """
    勝敗表示が出る画面中央付近を切り出す。
    """

    height, width = image.shape[:2]

    x1 = int(width * 0.20)
    x2 = int(width * 0.80)
    y1 = int(height * 0.30)
    y2 = int(height * 0.60)

    return image[y1:y2, x1:x2]


def load_templates(pattern):
    """
    templates フォルダから指定パターンのテンプレートを読み込む。
    """

    template_paths = sorted(
        Path("templates").glob(pattern)
    )

    templates = []

    for path in template_paths:
        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        edge_image = create_edge_image(
            image
        )

        templates.append(
            (
                path.name,
                edge_image,
            )
        )

    return templates


def match_template(source_edges, template_edges):
    """
    エッジ画像同士でテンプレートマッチングする。
    """

    source_height, source_width = source_edges.shape[:2]
    template_height, template_width = template_edges.shape[:2]

    if source_width < template_width:
        return 0.0

    if source_height < template_height:
        return 0.0

    result = cv2.matchTemplate(
        source_edges,
        template_edges,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_score, _, _ = cv2.minMaxLoc(result)

    return max_score


def get_best_score(source_edges, templates):
    """
    複数テンプレートの中で一番高いスコアを返す。
    """

    best_score = 0.0
    best_template_name = ""

    for template_name, template_edges in templates:
        score = match_template(
            source_edges,
            template_edges
        )

        if score > best_score:
            best_score = score
            best_template_name = template_name

    return best_score, best_template_name


def get_edge_scores(image):
    """
    WIN / LOSE のベストスコアと使用テンプレート名を返す。
    """

    roi = crop_result_area(image)

    roi_edges = create_edge_image(
        roi
    )

    win_templates = load_templates(
        "win*.png"
    )

    lose_templates = load_templates(
        "lose*.png"
    )

    if len(win_templates) == 0:
        return {
            "win_score": 0.0,
            "win_template": "",
            "lose_score": 0.0,
            "lose_template": "",
        }

    if len(lose_templates) == 0:
        return {
            "win_score": 0.0,
            "win_template": "",
            "lose_score": 0.0,
            "lose_template": "",
        }

    win_score, win_template = get_best_score(
        roi_edges,
        win_templates
    )

    lose_score, lose_template = get_best_score(
        roi_edges,
        lose_templates
    )

    return {
        "win_score": win_score,
        "win_template": win_template,
        "lose_score": lose_score,
        "lose_template": lose_template,
    }


def judge_result(win_score, lose_score):
    """
    スコアと差分から WIN / LOSE / NONE を判定する。
    """

    if win_score >= THRESHOLD and win_score - lose_score >= MARGIN:
        return "WIN"

    if lose_score >= THRESHOLD and lose_score - win_score >= MARGIN:
        return "LOSE"

    return "NONE"


def detect_result_by_edges(image):
    """
    エッジ検出と複数テンプレートを使って WIN / LOSE を判定する。
    """

    scores = get_edge_scores(image)

    return judge_result(
        scores["win_score"],
        scores["lose_score"]
    )