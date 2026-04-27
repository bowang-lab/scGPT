# Auto Issue Labeling and Claude Suggestions Setup

This document provides the GitHub Actions workflows for automatic issue labeling and Claude suggestions as requested in issue #325.

## Overview

Two workflows have been designed to automate issue management:

1. **Auto Issue Labeling** - Automatically categorizes and labels new issues
2. **Auto Issue Suggestions** - Provides helpful initial suggestions and guidance

## Installation Instructions

To enable these workflows, create the following files in your `.github/workflows/` directory:

### 1. Auto Issue Labeling Workflow

Create `.github/workflows/auto-issue-labeling.yml`:

```yaml
name: Auto Issue Labeling

on:
  issues:
    types: [opened]

jobs:
  auto-label:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      issues: write
      id-token: write
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Auto-label Issue
        uses: anthropics/claude-code-action@v1
        with:
          claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
          prompt: |
            You are helping with automatic issue labeling for the scGPT repository - a foundation model for single-cell multi-omics using generative AI.
            
            Please analyze this issue and suggest appropriate labels based on the content. Consider these categories:
            
            **Type labels:**
            - `bug` - Bug reports, errors, unexpected behavior
            - `enhancement` - Feature requests, improvements
            - `documentation` - Documentation improvements, clarifications
            - `question` - Questions about usage, implementation
            - `maintenance` - Code maintenance, refactoring, cleanup
            - `ci/cd` - CI/CD, testing, automation related
            
            **Component labels:**
            - `model` - Related to the core scGPT model architecture
            - `preprocessing` - Data preprocessing, tokenization
            - `training` - Training, fine-tuning workflows  
            - `inference` - Model inference, prediction
            - `integration` - Data integration tasks
            - `annotation` - Cell type annotation
            - `perturbation` - Perturbation analysis
            - `grn` - Gene regulatory network analysis
            - `tutorial` - Tutorial or example related
            - `installation` - Installation, setup issues
            
            **Priority labels:**
            - `priority-high` - Critical issues, security, major bugs
            - `priority-medium` - Important features, moderate bugs
            - `priority-low` - Nice to have, minor issues
            
            **Other labels:**
            - `good first issue` - Good for new contributors
            - `help wanted` - Community help requested
            - `duplicate` - Duplicate of existing issue
            - `wontfix` - Won't be implemented/fixed
            
            Based on the issue title and body, suggest 2-4 appropriate labels and apply them using:
            `gh issue edit ${{ github.event.issue.number }} --add-label "label1,label2,label3"`
            
            Be conservative with priority labels - only use priority-high for security issues or critical bugs that break core functionality.
          
          claude_args: '--allowed-tools "Bash(gh issue edit:*),Bash(gh issue view:*)"'
```

### 2. Auto Issue Suggestions Workflow

Create `.github/workflows/auto-issue-suggestions.yml`:

```yaml
name: Auto Issue Suggestions

on:
  issues:
    types: [opened]

jobs:
  auto-suggest:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      issues: write
      id-token: write
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Auto-suggest Solutions
        uses: anthropics/claude-code-action@v1
        with:
          claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
          prompt: |
            You are helping with automatic issue analysis for the scGPT repository - a foundation model for single-cell multi-omics using generative AI.
            
            Please analyze this new issue and provide helpful initial suggestions. Follow this approach:
            
            **For Bug Reports:**
            - Acknowledge the issue
            - Ask for reproduction steps if missing
            - Suggest checking common issues (installation, dependencies, data format)
            - Point to relevant documentation or tutorials
            - Suggest potential workarounds if applicable
            
            **For Feature Requests:**
            - Acknowledge the request
            - Ask clarifying questions about the use case
            - Suggest existing alternatives if available
            - Discuss implementation complexity and timeline considerations
            - Reference related issues or PRs if they exist
            
            **For Questions:**
            - Provide helpful guidance based on available documentation
            - Point to relevant tutorials, examples, or documentation sections
            - Suggest checking existing issues for similar questions
            - Offer next steps for getting help
            
            **For Documentation Issues:**
            - Acknowledge the documentation gap
            - Suggest interim solutions or resources
            - Ask for clarification on what specific information would be helpful
            
            **General Guidelines:**
            - Be welcoming and helpful
            - Reference specific documentation, tutorials, or code examples when relevant
            - Use the repository's knowledge (single-cell genomics, transformers, PyTorch)
            - Mention relevant files or directories when appropriate
            - Keep suggestions practical and actionable
            - Add a note that maintainers will review and provide additional guidance
            
            Add your suggestions as a comment using:
            `gh issue comment ${{ github.event.issue.number }} --body "Your helpful comment here"`
            
            Start your comment with: "👋 Thanks for opening this issue! Here are some initial suggestions..."
          
          claude_args: '--allowed-tools "Bash(gh issue comment:*),Bash(gh issue view:*)"'
```

## Setup Requirements

1. Ensure the `CLAUDE_CODE_OAUTH_TOKEN` secret is configured in your repository settings
2. Create the necessary labels in your repository if they don't exist:
   - Type labels: `bug`, `enhancement`, `documentation`, `question`, `maintenance`, `ci/cd`
   - Component labels: `model`, `preprocessing`, `training`, `inference`, `integration`, `annotation`, `perturbation`, `grn`, `tutorial`, `installation`
   - Priority labels: `priority-high`, `priority-medium`, `priority-low`
   - Other labels: `good first issue`, `help wanted`, `duplicate`, `wontfix`

## How It Works

1. **Auto Labeling**: When a new issue is opened, Claude analyzes the title and body to automatically apply relevant labels based on content, making issues easier to categorize and triage.

2. **Auto Suggestions**: Simultaneously, Claude provides an initial helpful comment with guidance, troubleshooting steps, or relevant documentation links based on the issue type.

## Benefits

- **Improved Issue Triage**: Automatic labeling helps maintainers quickly identify and prioritize issues
- **Better User Experience**: Users receive immediate helpful guidance instead of waiting for maintainer response
- **Consistent Responses**: Standardized approach to common issue types
- **Community Engagement**: Welcoming automated responses encourage community participation

## Customization

You can customize the label categories, priorities, and response templates by modifying the prompt sections in each workflow file to match your project's specific needs.