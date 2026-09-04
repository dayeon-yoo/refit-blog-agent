from orchestrator.workflow import run_workflow


def test_workflow_generates_top3_and_blog_post():
    result = run_workflow()

    assert "ideas" in result
    assert "top_3" in result
    assert "blog_posts" in result
    assert len(result["ideas"]) >= 10
    assert len(result["top_3"]) == 3
    assert len(result["blog_posts"]) == 3

    for post in result["blog_posts"]:
        assert post.title
        assert post.keyword
        assert post.content
        assert post.summary
        assert post.cta
