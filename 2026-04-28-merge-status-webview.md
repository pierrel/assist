# Add Merge Status to Web View List Items

## Reason for change
The web view currently shows thread status (busy stages, errors) but doesn't indicate merge status. Users should be able to see at a glance whether a thread has changes that can be merged, if there are merge conflicts, or if a merge is in progress.

## Proposed tests
1. Test that merge status badges are displayed correctly in the index view
2. Test that error status takes precedence over processing status
3. Test that merge status is properly retrieved from DomainManager
4. Test that badges are displayed in the correct order (error > processing > success)

## Proposed changes
1. Modify `render_index()` function in `/workspace/manage/web.py` to:
   - Check if a thread has a domain manager with merge status info
   - Add appropriate merge status badges next to list items
   - Ensure error status takes precedence over processing status
2. Add helper functions to determine merge status for a thread

## Expected outcomes
- Merge status badges appear next to each thread in the web view list
- Error status takes precedence over processing status
- Users can quickly identify threads that need attention for merging
- Existing functionality remains unchanged

## Risks / considerations
- Need to ensure performance impact is minimal
- Should not break existing UI when no domain manager is available
- Badge styling should be consistent with existing status indicators
- Error handling should be robust for threads without git repositories