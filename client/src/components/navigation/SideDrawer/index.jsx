import React from 'react';

import Backdrop from 'components/Backdrop';
import NavigationLink from '../NavigationLink';

import styles from './index.module.css';

const sideDrawer = props => {
    let classes = [styles.SideDrawer];
    if (props.isShown) {
        classes.push(styles.Open);
    } else {
        classes.push(styles.Close);
    }

    return (
        <>
            <Backdrop click={props.close} isShown={props.isShown} />
            <nav className={classes.join(' ')}>
                <ul className={styles.NavigationLinks}>
                    {props.links.map(link => (
                        <NavigationLink key={link.to} to={link.to}>
                            {link.value}
                        </NavigationLink>
                    ))}
                </ul>
            </nav>
        </>
    )
};

export default sideDrawer;