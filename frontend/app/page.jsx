import { HomePage } from '@/components/HomePage';

export const metadata = {
  title: "Atlas: Alzheimer's Knowledge Graph",
  description:
    "An ontology-grounded Graph RAG interface for Alzheimer's disease research. Ask questions in plain English; every answer traces to a node in a curated biomedical knowledge graph.",
};

export default function Page() {
  return <HomePage />;
}
