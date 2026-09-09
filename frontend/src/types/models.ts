export interface IdeaCandidate {
  candidate_id: string; title: string; perspective: string; content_format: string;
  key_question: string; brief_description: string;
}
export interface IdeaExpansionResult { source_input: string; candidates: IdeaCandidate[] }
export interface BlogIdea {
  title: string; keyword: string; search_intent: string; angle: string;
  rifit_connection: string; seasonality: number; summary: string;
  content_format?: string | null; content_perspective?: string | null;
  key_question?: string | null; target_reader?: string | null; outline?: string[] | null;
}
export interface ImagePlan {
  images: { placement: string; purpose: string; prompt: string; image_type: string; alt_text: string }[];
}
export interface BlogPost {
  title: string; keyword: string; content: string; summary: string; cta: string;
  tags?: string[] | null; image_plan?: ImagePlan | null;
}
export interface QCResult {
  status: 'PASS' | 'NEEDS_REVISION' | 'BLOCK'; summary: string;
  issues: { category: string; severity: 'warning' | 'error'; message: string; evidence: string; suggestion: string }[];
}
export interface WorkflowResult {
  blog_idea: BlogIdea; content_plan: { writing_script: string }; post: BlogPost;
  tags: string[]; image_plan: ImagePlan | null; qc: QCResult;
}
