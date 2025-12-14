/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 *
 * Original work:
 *   Copyright (C) 2022–2024 Henrique Fingler
 *   Copyright (C) 2022–2024 Isha Tarte
 *
 * Modifications and adaptations for LAIKA:
 *   Copyright (C) 2024-2025 Haoming Zhuo
 *
 * This file is adapted from the original LAKE kernel module.
 * Major changes include:
 *   - Integration with LAIKA framework
 *   - Hybrid execution support
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <linux/dcache.h>
#include <linux/namei.h>
#include <linux/mount.h>
#include <linux/fs_stack.h>
#include <linux/slab.h>
#include "ecryptfs_kernel.h"

/**
 * ecryptfs_d_revalidate - revalidate an ecryptfs dentry
 * @dentry: The ecryptfs dentry
 * @flags: lookup flags
 *
 * Called when the VFS needs to revalidate a dentry. This
 * is called whenever a name lookup finds a dentry in the
 * dcache. Most filesystems leave this as NULL, because all their
 * dentries in the dcache are valid.
 *
 * Returns 1 if valid, 0 otherwise.
 *
 */
static int ecryptfs_d_revalidate(struct dentry *dentry, unsigned int flags)
{
	struct dentry *lower_dentry = ecryptfs_dentry_to_lower(dentry);
	int rc = 1;

	if (flags & LOOKUP_RCU)
		return -ECHILD;

	if (lower_dentry->d_flags & DCACHE_OP_REVALIDATE)
		rc = lower_dentry->d_op->d_revalidate(lower_dentry, flags);

	if (d_really_is_positive(dentry)) {
		struct inode *inode = d_inode(dentry);

		fsstack_copy_attr_all(inode, ecryptfs_inode_to_lower(inode));
		if (!inode->i_nlink)
			return 0;
	}
	return rc;
}

struct kmem_cache *ecryptfs_dentry_info_cache;

static void ecryptfs_dentry_free_rcu(struct rcu_head *head)
{
	kmem_cache_free(ecryptfs_dentry_info_cache,
		container_of(head, struct ecryptfs_dentry_info, rcu));
}

/**
 * ecryptfs_d_release
 * @dentry: The ecryptfs dentry
 *
 * Called when a dentry is really deallocated.
 */
static void ecryptfs_d_release(struct dentry *dentry)
{
	struct ecryptfs_dentry_info *p = dentry->d_fsdata;
	if (p) {
		path_put(&p->lower_path);
		call_rcu(&p->rcu, ecryptfs_dentry_free_rcu);
	}
}

const struct dentry_operations ecryptfs_dops = {
	.d_revalidate = ecryptfs_d_revalidate,
	.d_release = ecryptfs_d_release,
};
